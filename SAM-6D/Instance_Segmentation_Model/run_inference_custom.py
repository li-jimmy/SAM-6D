import os, sys
import numpy as np
import shutil
from tqdm import tqdm
import time
import torch
from PIL import Image
import logging
import os, sys
import os.path as osp
from hydra import initialize, compose
# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import trimesh
import numpy as np
from hydra.utils import instantiate
import argparse
import glob
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
import torchvision.transforms as T
import cv2
import imageio
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask
from model.detector import Instance_Segmentation_Model
from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from utils.bbox_utils import CropResizePad
from model.utils import Detections, convert_npz_to_json
from model.loss import Similarity
from utils.inout import load_json, save_json_bop23
from utils.bbox_utils import force_binary_mask

inv_rgb_transform = T.Compose(
        [
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ]
    )

def visualize(rgb, detections, save_path="tmp.png"):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33

    best_score = 0.
    for mask_idx, det in enumerate(detections):
        if best_score < det['score']:
            best_score = det['score']
            best_det = detections[mask_idx]

    mask = rle_to_mask(best_det["segmentation"])
    edge = canny(mask)
    edge = binary_dilation(edge, np.ones((2, 2)))
    obj_id = best_det["category_id"]
    temp_id = obj_id - 1

    r = int(255*colors[temp_id][0])
    g = int(255*colors[temp_id][1])
    b = int(255*colors[temp_id][2])
    img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
    img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
    img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
    img[edge, :] = 255
    
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)
    
    # concat side by side in PIL
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat

def visualize_one(rgb, mask, save_path="tmp.png"):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    colors = distinctipy.get_colors(1)
    alpha = 0.33

    #best_score = 0.
    # for mask_idx, det in enumerate(detections):
    #     if best_score < det['score']:
    #         best_score = det['score']
    #         best_det = detections[mask_idx]

    # mask = rle_to_mask(best_det["segmentation"])
    edge = canny(mask)
    edge = binary_dilation(edge, np.ones((2, 2)))
    # obj_id = best_det["category_id"]
    # temp_id = obj_id - 1

    r = int(255*colors[0][0])
    g = int(255*colors[0][1])
    b = int(255*colors[0][2])
    img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
    img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
    img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
    img[edge, :] = 255
    
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)
    
    # concat side by side in PIL
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat

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

def run_inference(segmentor_model, output_dir, cad_path, rgb_path, depth_path, cam_path, stability_score_thresh):
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name='run_inference.yaml')

    if segmentor_model == "sam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_sam.yaml')
        cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
    elif segmentor_model == "fastsam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_fastsam.yaml')
    else:
        raise ValueError("The segmentor_model {} is not supported now!".format(segmentor_model))

    logging.info("Initializing model")
    model: Instance_Segmentation_Model = instantiate(cfg.model)
    
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
    template_dir = args.template_path
    num_templates = len(glob.glob(f"{template_dir}/*.txt"))
    print(' - Num templates:', num_templates)
    boxes, masks, templates, tmpl_poses = [], [], [], []
    for idx in range(num_templates):
        image = Image.open(os.path.join(template_dir, 'rgb_'+str(idx)+'.png'))
        mask = Image.open(os.path.join(template_dir, 'mask_'+str(idx)+'.png'))
        pose = np.loadtxt(os.path.join(template_dir, 'pose_'+str(idx)+'.txt'))
        tmpl_poses.append(pose)
        boxes.append(mask.getbbox())

        image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
        mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
        image = image * mask[:, :, None]
        templates.append(image)
        masks.append(mask.unsqueeze(-1))
        
    templates = torch.stack(templates).permute(0, 3, 1, 2)
    masks = torch.stack(masks).permute(0, 3, 1, 2)
    tmpl_poses = torch.stack([torch.tensor(pose) for pose in tmpl_poses])
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

    print ('Inference starting')
    start_time = time.time()

    # run inference
    rgb = Image.open(rgb_path).convert("RGB")
    detections = model.segmentor_model.generate_masks(np.array(rgb))
    detections = Detections(detections)
    query_decriptors, query_appe_descriptors = model.descriptor_model.forward(np.array(rgb), detections)

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
    batch = batch_input_data(depth_path, cam_path, device)
    # template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
    # template_poses[:, :3, 3] *= 0.4
    poses = torch.tensor(tmpl_poses).to(torch.float32).to(device)
    #model.ref_data["poses"] =  poses[load_index_level_in_level2(0, "all"), :, :]
    model.ref_data["poses"] = poses
    #model.ref_data["cam_poses"] = np.load('../Instance_Segmentation_Model/utils/poses/predefined_poses/cam_poses_level0.npy')
    mesh = trimesh.load_mesh(cad_path)
    model_points = mesh.sample(2048).astype(np.float32) # / 1000.0
    model.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(device)
    detections.masks.squeeze_()
    image_uv = model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)
    
    geometric_score, visible_ratio = model.compute_geometric_score(
        image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=model.visible_thred
    )
    visible_ratio = model.compute_visible_ratio(query_appe_descriptors, ref_aux_descriptor, visible_thred=model.visible_thred)

    # final score
    #final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)
    final_score = (semantic_score + appe_scores) * visible_ratio

    detections.add_attribute("scores", final_score)
    detections.add_attribute("object_ids", torch.zeros_like(final_score))   
    
    detections.add_attribute("semantic_score", semantic_score)
    detections.add_attribute("appe_scores", appe_scores)
    detections.add_attribute("geometric_score", geometric_score)
    detections.add_attribute("visible_ratio", visible_ratio)
    detections.add_attribute("best_template", best_template)

    detections.to_numpy()
    #top5_idxs = np.argsort(detections.scores)[-5:][::-1]
    all_sorted_idxs = np.argsort(detections.scores)[::-1]
    print ('Inference finished')
    print ('Time taken:', time.time()-start_time)
    save_path = f"{output_dir}/sam6d_results/detection_ism"
    for i, idx in enumerate(all_sorted_idxs):
        if detections.semantic_score[idx] < 0.5 or detections.visible_ratio[idx] < 0.5:
            continue
        print('Rank', i, 'Idx', idx)
        print(' - Score', detections.scores[idx])
        print(' - Semantic score', detections.semantic_score[idx])
        print(' - Appearance score', detections.appe_scores[idx])
        print(' - Geometric score', detections.geometric_score[idx])
        print(' - Visible ratio', detections.visible_ratio[idx])
        print(' - Best template', detections.best_template[idx])
        print(' - Best template pose', model.ref_data["poses"][detections.best_template[idx]])
        
        print(' - ACCEPTED')
        binary_mask = force_binary_mask(detections.masks[idx]).astype(np.uint8)
        mask_save_path = f"{save_path}_mask_{i}.png"
        Image.fromarray(binary_mask * 255).save(mask_save_path)

        inverted_mask = 1 - binary_mask
        inverted_mask_color = cv2.cvtColor(inverted_mask * 255, cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(np.array(rgb), 1, inverted_mask_color, 0.8, 0)
        overlay_save_path = f"{save_path}_overlay_{i}.png"
        Image.fromarray(overlay).save(overlay_save_path)
        
        best_template_path = os.path.join(template_dir, 'rgb_'+str(detections.best_template[idx])+'.png')
        shutil.copy(best_template_path, f"{save_path}_best_template_{i}.png")
        
        vis_pc = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
        image_uv1 = image_uv[idx]
        for point in image_uv1:
            x, y = int(point[0]), int(point[1])
            cv2.circle(vis_pc, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
        cv2.imwrite(f"{save_path}_pc_{i}.png", vis_pc)

        # vis_img = visualize_one(rgb, binary_mask, f"{save_path}_{i}.png")
        # vis_img.save(f"{save_path}_{i}.png")

        # Save the best mask, to be used downstream
        if i == 0:
            print('Copying best mask to', args.rgb_path.replace('_color', '_mask'))
            shutil.copy(mask_save_path, args.rgb_path.replace('_color', '_mask'))

        

    # detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)
    # detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
    # detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    # save_json_bop23(save_path+".json", detections)
    # vis_img = visualize(rgb, detections, f"{output_dir}/sam6d_results/vis_ism.png")
    # vis_img.save(f"{output_dir}/sam6d_results/vis_ism.png")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentor_model", default='sam', help="The segmentor model in ISM")
    parser.add_argument("--output_dir", nargs="?", help="Path to root directory of the output")
    parser.add_argument("--cad_path", nargs="?", help="Path to CAD(mm)")
    parser.add_argument("--rgb_path", nargs="?", help="Path to RGB image")
    parser.add_argument("--depth_path", nargs="?", help="Path to Depth image(mm)")
    parser.add_argument("--template_path", help="Path to templates")
    parser.add_argument("--cam_path", nargs="?", help="Path to camera information")
    parser.add_argument("--stability_score_thresh", default=0.97, type=float, help="stability_score_thresh of SAM")
    args = parser.parse_args()
    os.makedirs(f"{args.output_dir}/sam6d_results", exist_ok=True)
    run_inference(
        args.segmentor_model, args.output_dir, args.cad_path, args.rgb_path, args.depth_path, args.cam_path, 
        stability_score_thresh=args.stability_score_thresh,
    )