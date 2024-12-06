import asyncio
from contextlib import asynccontextmanager
import io
import os
import numpy as np
import shutil
import time
import torch
from PIL import Image
import logging
from hydra import initialize, compose
from hydra.utils import instantiate
import argparse
import glob
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
import torchvision.transforms as T
import cv2
from model.detector import Instance_Segmentation_Model
from utils.bbox_utils import CropResizePad
from model.utils import Detections
from utils.bbox_utils import force_binary_mask
from fastapi import FastAPI, File, UploadFile
from hypercorn.config import Config
from hypercorn.asyncio import serve
import httpx

# set level logging
logging.basicConfig(level=logging.WARN)

parser = argparse.ArgumentParser()
parser.add_argument("--segmentor_model", default='sam', help="The segmentor model in ISM")
parser.add_argument("--debug_dir", required=False, help="Path to root directory of the output")
parser.add_argument("--cad_path", required=True, help="Path to CAD(mm)")
parser.add_argument("--template_path", required=True, help="Path to templates")
parser.add_argument("--stability_score_thresh", default=0.97, type=float, help="stability_score_thresh of SAM")
parser.add_argument("--pose_server", default='http://localhost:48003/register/', type=str, help="URL for computing pose of detected object")
parser.add_argument('--ip',
                    type=str,
                    default='0.0.0.0')
parser.add_argument('--port',
                    type=int,
                    default=48001)
args = parser.parse_args()
    

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, http_client
    http_client = httpx.Client(http2=True, verify=False)  # `verify=False` for self-signed certificates

    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name='run_inference.yaml')

    if args.segmentor_model == "sam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_sam.yaml')
        cfg.model.segmentor_model.stability_score_thresh = args.stability_score_thresh
    elif args.segmentor_model == "fastsam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_fastsam.yaml')
    else:
        raise ValueError("The segmentor_model {} is not supported now!".format(args.segmentor_model))

    logging.info("Initializing model")
    model = instantiate(cfg.model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    # if there is predictor in the model, move it to device
    if hasattr(model.segmentor_model, "predictor"):
        model.segmentor_model.predictor.model = (
            model.segmentor_model.predictor.model.to(device)
        )
    else:
        model.segmentor_model.model.setup_model(device=device, verbose=True)
    logging.info(f"Moving models to {device} done!")
        
    logging.info("Initializing template")
    start_time = time.time()
    num_templates = len(glob.glob(os.path.join(args.template_path, "*.npy")))
    print(' - Num templates:', num_templates)
    boxes, masks, templates = [], [], []
    for idx in range(num_templates):
        image = Image.open(os.path.join(args.template_path, 'rgb_'+str(idx)+'.png'))
        mask = Image.open(os.path.join(args.template_path, 'mask_'+str(idx)+'.png'))
        boxes.append(mask.getbbox())

        image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
        mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
        image = image * mask[:, :, None]
        templates.append(image)
        masks.append(mask.unsqueeze(-1))
        
    templates = torch.stack(templates).permute(0, 3, 1, 2)
    masks = torch.stack(masks).permute(0, 3, 1, 2)
    boxes = torch.tensor(np.array(boxes))
    
    processing_config = OmegaConf.create(
        {
            "image_size": 224,
        }
    )
    proposal_processor = CropResizePad(processing_config.image_size)
    templates = proposal_processor(images=templates, boxes=boxes).to(device)
    masks_cropped = proposal_processor(images=masks, boxes=boxes).to(device)

    model.ref_data = {}
    model.ref_data["descriptors"] = model.descriptor_model.compute_features(
                    templates, token_name="x_norm_clstoken"
                ).unsqueeze(0).data
    model.ref_data["appe_descriptors"] = model.descriptor_model.compute_masked_patch_feature(
                    templates, masks_cropped[:, 0, :, :]
                ).unsqueeze(0).data
    
    print('Template processing time:', time.time()-start_time)
    
    yield
    http_client.close()

app = FastAPI(lifespan=lifespan)

@app.post("/segment_and_register/")
async def predict(rgb: UploadFile = File(...), depth: UploadFile = File(...)):
    start_time = time.time()

    rgb_bytes = await rgb.read()
    depth_bytes = await depth.read()
    rgb_array = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)
    
    detections = model.segmentor_model.generate_masks(rgb_array)
    detections = Detections(detections)
    query_decriptors, query_appe_descriptors = model.descriptor_model.forward(rgb_array, detections)

    # matching descriptors
    (
        idx_selected_proposals,
        pred_idx_objects,
        semantic_score,
        best_template,
    ) = model.compute_semantic_score(query_decriptors)

    # update detections
    detections.filter(idx_selected_proposals)
    query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

    # compute the appearance score
    appe_scores, ref_aux_descriptor= model.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

    # compute the geometric score
    # batch = batch_input_data(depth_path, cam_path, device)
    # template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
    # template_poses[:, :3, 3] *= 0.4
    # poses = torch.tensor(template_poses).to(torch.float32).to(device)
    # model.ref_data["poses"] =  poses[load_index_level_in_level2(0, "all"), :, :]

    # mesh = trimesh.load_mesh(cad_path)
    # model_points = mesh.sample(2048).astype(np.float32) / 1000.0
    # model.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(device)
    detections.masks.squeeze_()
    #image_uv = model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

    # geometric_score, visible_ratio = model.compute_geometric_score(
    #     image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=model.visible_thred
    #     )
    visible_ratio = model.compute_visible_ratio(query_appe_descriptors, ref_aux_descriptor, visible_thred=model.visible_thred)

    # final score
    #final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)
    final_score = (semantic_score + appe_scores) * visible_ratio

    detections.add_attribute("scores", final_score)
    detections.add_attribute("object_ids", torch.zeros_like(final_score))   
    
    detections.add_attribute("semantic_score", semantic_score)
    detections.add_attribute("appe_scores", appe_scores)
    #detections.add_attribute("geometric_score", geometric_score)
    detections.add_attribute("visible_ratio", visible_ratio)
    detections.add_attribute("best_template", best_template)

    detections.to_numpy()
    all_sorted_idxs = np.argsort(detections.scores)[::-1]
    print ('Inference took:', time.time()-start_time)
    
    for i, idx in enumerate(all_sorted_idxs):
        if detections.semantic_score[idx] < 0.5 or detections.visible_ratio[idx] < 0.5:
            continue
        
        binary_mask = force_binary_mask(detections.masks[idx]).astype(np.uint8)
        mask_image = Image.fromarray(binary_mask * 255)
        mask_bytes = io.BytesIO()
        mask_image.save(mask_bytes, format='PNG')
        mask_bytes.seek(0)
        
        if args.debug_dir is not None:
            save_path = os.path.join(args.debug_dir, 'detection_ism')
            mask_save_path = f"{save_path}_mask.png"
            Image.fromarray(binary_mask * 255).save(mask_save_path)

            inverted_mask = 1 - binary_mask
            inverted_mask_color = cv2.cvtColor(inverted_mask * 255, cv2.COLOR_GRAY2RGB)
            overlay = cv2.addWeighted(rgb_array, 1, inverted_mask_color, 0.8, 0)
            overlay_save_path = f"{save_path}_overlay.png"
            Image.fromarray(overlay).save(overlay_save_path)
            best_template_path = os.path.join(args.template_path, 'rgb_'+str(detections.best_template[idx])+'.png')
            shutil.copy(best_template_path, f"{save_path}_best_template.png")
            
        files = {
            "rgb": ("rgb_image.png", rgb_bytes, "image/png"),
            "depth": ("depth_image.png", depth_bytes, "image/png"),
            "mask": ("mask_image.png", mask_bytes, "image/png")
        }
        
        response = http_client.post(args.pose_server, files=files)

        # Check and print the response
        if response.status_code == 200:
            return response.json()
        else:
            return {'success': False, 'details': 'Pose estimation failed'}
    return {'success': False, 'details': 'No object detected'}

if __name__ == "__main__":
    #uvicorn.run(app, host=args.ip, port=args.port)
    config = Config()
    config.bind = [f"{args.ip}:{args.port}"] 
    config.http2 = True
    asyncio.run(serve(app, config))