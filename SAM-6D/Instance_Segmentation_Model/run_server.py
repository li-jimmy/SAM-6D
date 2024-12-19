import asyncio
import base64
from contextlib import asynccontextmanager
import io
import json
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
from utils.bbox_utils import force_binary_mask, force_binary_mask_torch
from fastapi import FastAPI, File, UploadFile
from hypercorn.config import Config
from hypercorn.asyncio import serve
import httpx
from utils.inout import load_json
import imageio
from fastapi.responses import StreamingResponse

# set level logging
logging.basicConfig(level=logging.WARN)

parser = argparse.ArgumentParser()
parser.add_argument("--segmentor_model", default='fastsam', help="The segmentor model in ISM")
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

def batch_input_data(depth_path, cam_path, device):
    batch = {}
    cam_info = load_json(cam_path)
    depth = np.array(imageio.imread(depth_path)).astype(np.int32)
    cam_K = np.array(cam_info['cam_K']).reshape((3, 3))
    depth_scale = np.array(cam_info['depth_scale'])

    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, http_client, tmpl_poses
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
    num_templates = len(glob.glob(os.path.join(args.template_path, "pose_*.txt")))
    print(' - Num templates:', num_templates)
    boxes, masks, templates, tmpl_poses = [], [], [], []
    for idx in range(num_templates):
        image = Image.open(os.path.join(args.template_path, 'rgb_'+str(idx)+'.png'))
        mask = Image.open(os.path.join(args.template_path, 'mask_'+str(idx)+'.png'))
        pose = np.loadtxt(os.path.join(args.template_path, 'pose_'+str(idx)+'.txt'))
        tmpl_poses.append(pose)
        boxes.append(mask.getbbox())

        image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
        mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
        image = image * mask[:, :, None]
        templates.append(image)
        masks.append(mask.unsqueeze(-1))
        
    templates = torch.stack(templates).permute(0, 3, 1, 2)
    masks = torch.stack(masks).permute(0, 3, 1, 2)
    boxes = torch.tensor(np.array(boxes))
    tmpl_poses = torch.stack([torch.tensor(pose) for pose in tmpl_poses])
    
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

def depth_filter(detections, rgb_array, depth_array):
    start_time = time.time()
    depth_array = depth_array.astype(np.float64) / 1000.0
    depth_array[(depth_array<0.001) | (depth_array>=np.inf)] = 0
    avg_depth_values = []
    masks_np = detections.masks.cpu().numpy()
    for det_i in range(len(detections.masks)):
        mask = masks_np[det_i]
        binary_mask = force_binary_mask(mask).astype(bool)
        depth_values = depth_array[binary_mask]
        non_zero_depths = depth_values[depth_values > 0]
        if len(non_zero_depths) > 0:
            average_depth = np.mean(non_zero_depths)
            avg_depth_values.append(average_depth)
        else:
            avg_depth_values.append(np.inf)
    print('Depth filtering took:', time.time()-start_time)
    depth_sort = np.argsort(avg_depth_values)
    for i, det_i in enumerate(depth_sort):
        mask = masks_np[det_i]
        binary_mask = force_binary_mask(mask).astype(np.uint8)
        binary_mask_save_path = f"{args.debug_dir}/dist_sort/mask_{i}_{det_i}.png"
        Image.fromarray(binary_mask * 255).save(binary_mask_save_path)
        inverted_mask = 1 - binary_mask
        inverted_mask_color = cv2.cvtColor(inverted_mask * 255, cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(rgb_array, 1, inverted_mask_color, 0.8, 0)
        overlay_save_path = f"{args.debug_dir}/dist_sort/viz_{i}_{det_i}.png"
        Image.fromarray(overlay).save(overlay_save_path)
    return detections

def size_filter(detections, min_size=30000):
    binary_masks = force_binary_mask_torch(detections.masks)
    mask_sizes = binary_masks.sum(dim=(1, 2))
    selected_indices = (mask_sizes > min_size).nonzero(as_tuple=True)[0]
    detections.filter(selected_indices)
    return detections

def segment(rgb_array):
    torch.cuda.synchronize()
    start_time0 = time.time()
    start_time = time.time()
    detections = model.segmentor_model.generate_masks(rgb_array)
    detections = Detections(detections)
    torch.cuda.synchronize()
    print('Segmentation took:', time.time()-start_time)
    print('Number of raw segments:', len(detections))
    detections = size_filter(detections)
    print('Number of segments after size filtering:', len(detections))
    start_time = time.time()
    query_decriptors, query_appe_descriptors = model.descriptor_model.forward(rgb_array, detections)
    torch.cuda.synchronize()
    print('Descriptor computation took:', time.time()-start_time)
    
    start_time = time.time()
    # matching descriptors
    (
        idx_selected_proposals,
        pred_idx_objects,
        semantic_score,
        best_template,
    ) = model.compute_semantic_score(query_decriptors)
    torch.cuda.synchronize()
    print('Semantic scoring took:', time.time()-start_time)
    
    # update detections
    detections.filter(idx_selected_proposals)
    query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]
    
    start_time = time.time()
    # compute the appearance score
    appe_scores, ref_aux_descriptor= model.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)
    torch.cuda.synchronize()
    print('Appearance scoring took:', time.time()-start_time)


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
    start_time = time.time()
    visible_ratio = model.compute_visible_ratio(query_appe_descriptors, ref_aux_descriptor, visible_thred=model.visible_thred)
    torch.cuda.synchronize()
    print('Visibility ratio took:', time.time()-start_time)
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
    
    for i, idx in enumerate(all_sorted_idxs):
        if detections.semantic_score[idx] < 0.5 or detections.visible_ratio[idx] < 0.5:
            continue
        
        start_time = time.time()
        best_ori_template = model.estimate_orientation(pred_idx_objects[idx], query_appe_descriptors[idx])
        best_ori_template_pose = tmpl_poses[best_ori_template]
        print('Coarse pose estimate:')
        print(best_ori_template_pose)
        print('Orientation estimation took:', time.time()-start_time)
    
        binary_mask = force_binary_mask(detections.masks[idx]).astype(np.uint8)
        print ('Instance segmentation total time:', time.time()-start_time0)
        
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
            best_ori_template_path = os.path.join(args.template_path, 'rgb_'+str(best_ori_template)+'.png')
            shutil.copy(best_ori_template_path, f"{save_path}_best_ori_template.png")
            
        return binary_mask, best_ori_template_pose.cpu().numpy()
    return None, None

@app.post("/segment/")
async def segment_endpoint(rgb: UploadFile = File(...)):
    rgb_bytes = await rgb.read()
    rgb_array = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)
    
    binary_mask, pose_estimate = segment(rgb_array)
    if binary_mask is not None:
        mask_image = Image.fromarray(binary_mask * 255)
        mask_bytes = io.BytesIO()
        mask_image.save(mask_bytes, format='PNG')
        mask_bytes.seek(0)
        return StreamingResponse(mask_bytes, media_type="image/png")
    return {'success': False, 'details': 'No object detected'}

@app.post("/segment_and_register/")
async def segment_and_register_endpoint(rgb: UploadFile = File(...), depth: UploadFile = File(...)):
    start_time = time.time()
    rgb_bytes = await rgb.read()
    depth_bytes = await depth.read()
    rgb_array = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)
    #depth_array = cv2.imdecode(np.frombuffer(depth_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    
    binary_mask, pose_estimate = segment(rgb_array)
    if binary_mask is not None:
        mask_image = Image.fromarray(binary_mask * 255)
        mask_bytes = io.BytesIO()
        mask_image.save(mask_bytes, format='PNG')
        mask_bytes.seek(0)
        
        files = {
            "rgb": ("rgb_image.png", rgb_bytes, "image/png"),
            "depth": ("depth_image.png", depth_bytes, "image/png"),
            "mask": ("mask_image.png", mask_bytes, "image/png")
        }
        
        pose_estimate_str = json.dumps(pose_estimate.tolist())
        
        #response = http_client.post(args.pose_server, files=files, data={'coarse_estimate': pose_estimate_str})
        response = http_client.post(args.pose_server, files=files, data={'coarse_estimate': ''})

        # Check and print the response
        if response.status_code == 200:
            resp = response.json()
            resp['success'] = True
            print('Segment and register total time:', time.time()-start_time)
            return resp
        else:
            return {'success': False, 'details': 'Pose estimation failed'}
    return {'success': False, 'details': 'No object detected'}

if __name__ == "__main__":
    config = Config()
    config.bind = [f"{args.ip}:{args.port}"] 
    config.http2 = True
    asyncio.run(serve(app, config))