import cv2
import torch
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
from PIL import Image
import logging
import os
import os.path as osp
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from utils.inout import save_json, load_json, save_json_bop23
from model.utils import BatchedData, Detections, convert_npz_to_json
from hydra.utils import instantiate
import time
import glob
from functools import partial
import multiprocessing
import trimesh
from model.loss import MaskedPatch_MatrixSimilarity
from utils.trimesh_utils import depth_image_to_pointcloud_translate_torch
from utils.poses.pose_utils import get_obj_poses_from_template_level
from utils.bbox_utils import xyxy_to_xywh, compute_iou
from scipy.spatial.transform import Rotation

class Instance_Segmentation_Model(pl.LightningModule):
    def __init__(
        self,
        segmentor_model,
        descriptor_model,
        onboarding_config,
        matching_config,
        post_processing_config,
        log_interval,
        log_dir,
        visible_thred,
        pointcloud_sample_num,
        **kwargs,
    ):
        # define the network
        super().__init__()
        self.segmentor_model = segmentor_model
        self.descriptor_model = descriptor_model

        self.onboarding_config = onboarding_config
        self.matching_config = matching_config
        self.post_processing_config = post_processing_config
        self.log_interval = log_interval
        self.log_dir = log_dir

        self.visible_thred = visible_thred
        self.pointcloud_sample_num = pointcloud_sample_num

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(osp.join(self.log_dir, "predictions"), exist_ok=True)
        self.inv_rgb_transform = T.Compose(
            [
                T.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                ),
            ]
        )
        logging.info(f"Init CNOS done!")

    def set_reference_objects(self):
        os.makedirs(
            osp.join(self.log_dir, f"predictions/{self.dataset_name}"), exist_ok=True
        )
        logging.info("Initializing reference objects ...")

        start_time = time.time()
        self.ref_data = {
            "descriptors": BatchedData(None),
            "appe_descriptors": BatchedData(None),
                         }
        descriptors_path = osp.join(self.ref_dataset.template_dir, "descriptors.pth")
        appe_descriptors_path = osp.join(self.ref_dataset.template_dir, "descriptors_appe.pth")

        # Loading main descriptors
        if self.onboarding_config.rendering_type == "pbr":
            descriptors_path = descriptors_path.replace(".pth", "_pbr.pth")
        if (
            os.path.exists(descriptors_path)
            and not self.onboarding_config.reset_descriptors
        ):
            self.ref_data["descriptors"] = torch.load(descriptors_path).to(self.device)
        else:
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Computing descriptors ...",
            ):
                ref_imgs = self.ref_dataset[idx]["templates"].to(self.device)
                ref_feats = self.descriptor_model.compute_features(
                    ref_imgs, token_name="x_norm_clstoken"
                )
                self.ref_data["descriptors"].append(ref_feats)

            self.ref_data["descriptors"].stack()  # N_objects x descriptor_size
            self.ref_data["descriptors"] = self.ref_data["descriptors"].data

            # save the precomputed features for future use
            torch.save(self.ref_data["descriptors"], descriptors_path)

        # Loading appearance descriptors
        if self.onboarding_config.rendering_type == "pbr":
            appe_descriptors_path = appe_descriptors_path.replace(".pth", "_pbr.pth")
        if (
            os.path.exists(appe_descriptors_path)
            and not self.onboarding_config.reset_descriptors
        ):
            self.ref_data["appe_descriptors"] = torch.load(appe_descriptors_path).to(self.device)
        else:
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Computing appearance descriptors ...",
            ):
                ref_imgs = self.ref_dataset[idx]["templates"].to(self.device)
                ref_masks = self.ref_dataset[idx]["template_masks"].to(self.device)
                ref_feats = self.descriptor_model.compute_masked_patch_feature(
                    ref_imgs, ref_masks
                )
                self.ref_data["appe_descriptors"].append(ref_feats)

            self.ref_data["appe_descriptors"].stack()
            self.ref_data["appe_descriptors"] = self.ref_data["appe_descriptors"].data

            # save the precomputed features for future use
            torch.save(self.ref_data["appe_descriptors"], appe_descriptors_path)

        end_time = time.time()
        logging.info(
            f"Runtime: {end_time-start_time:.02f}s, Descriptors shape: {self.ref_data['descriptors'].shape}, \
            Appearance descriptors shape: {self.ref_data['appe_descriptors'].shape}"
        )

    def set_reference_object_pointcloud(self):
        """
            Loading the pointclouds of reference objects: (N_object, N_pointcloud, 3)
            N_pointcloud: the number of points sampled from the reference object mesh.
        """
        os.makedirs(
            osp.join(self.log_dir, f"predictions/{self.dataset_name}"), exist_ok=True
        )
        logging.info("Initializing reference objects point cloud ...")
        
        start_time = time.time()
        pointcloud = BatchedData(None)
        pointcloud_path = osp.join(self.ref_dataset.template_dir, "pointcloud.pth")
        obj_pose_path = f"{self.ref_dataset.template_dir}/template_poses.npy"

        # Loading pointcloud pose
        if (
            os.path.exists(obj_pose_path)
            and not self.onboarding_config.reset_descriptors
        ):
            poses = torch.tensor(np.load(obj_pose_path)).to(self.device).to(torch.float32) # N_all_template x 4 x 4
        else:
            template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
            template_poses[:, :3, 3] *= 0.4
            poses = torch.tensor(template_poses).to(self.device).to(torch.float32)
            np.save(obj_pose_path, template_poses)
        
        self.ref_data["poses"] = poses[self.ref_dataset.index_templates, :, :] # N_template x 4 x 4
        if (
            os.path.exists(pointcloud_path)
            and not self.onboarding_config.reset_descriptors
        ):
            self.ref_data["pointcloud"] = torch.load(pointcloud_path, map_location="cuda:0").to(self.device)
        else:
            mesh_path = osp.join(self.ref_dataset.root_dir, "models")
            if not os.path.exists(mesh_path):
                raise Exception("Can not find the mesh path.")
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Generating pointcloud dataset ...",
            ):
                # loading cad
                if self.dataset_name == "lmo":
                    all_pc_idx = [1, 5, 6, 8, 9, 10, 11, 12]
                    pc_id = all_pc_idx[idx]
                else:
                    pc_id = idx + 1
                mesh = trimesh.load_mesh(os.path.join(mesh_path, f'obj_{(pc_id):06d}.ply'))
                model_points = mesh.sample(self.pointcloud_sample_num).astype(np.float32) / 1000.0
                pointcloud.append(torch.tensor(model_points))

            pointcloud.stack()  # N_objects x N_pointcloud x 3
            self.ref_data["pointcloud"] = pointcloud.data.to(self.device)

            # save the precomputed features for future use
            torch.save(self.ref_data["pointcloud"], pointcloud_path)

        end_time = time.time()
        logging.info(
            f"Runtime: {end_time-start_time:.02f}s, Pointcloud shape: {self.ref_data['pointcloud'].shape}"
        )

    def best_template_pose(self, scores, pred_idx_objects):
        _, best_template_idxes = torch.max(scores, dim=-1)
        N_query, N_object = best_template_idxes.shape[0], best_template_idxes.shape[1]
        pred_idx_objects = pred_idx_objects[:, None].repeat(1, N_object)

        assert N_query == pred_idx_objects.shape[0], "Prediction num != Query num"

        best_template_idx = torch.gather(best_template_idxes, dim=1, index=pred_idx_objects)[:, 0]

        return best_template_idx

    def project_templates(self):
        cam_poses = self.ref_data["cam_poses"].copy()
        obj_poses = []
        for cam_pose in cam_poses:
            cam_pose[:3, 1:3] = -cam_pose[:3, 1:3]
            obj_poses.append(np.linalg.inv(cam_pose))
        # cam_poses = torch.tensor(cam_poses, dtype=torch.float32).to('cuda')
        # obj_poses = torch.tensor(obj_poses, dtype=torch.float32).to('cuda')
        obj_poses = np.array(obj_poses)
        pose_R_blender = obj_poses[:, 0:3, 0:3]
        #pose_R[:3, 1:3] = -pose_R[:3, 1:3]
        # Convert Blender's coordinate system to OpenCV's coordinate system
        # Blender: +X right, +Y forward, +Z up
        # OpenCV: +X right, +Y down, +Z forward
        blender_to_opencv = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        pose_R = blender_to_opencv @ pose_R_blender @ blender_to_opencv
        import ipdb; ipdb.set_trace()
        pose_R = torch.tensor(pose_R, dtype=torch.float32).to('cuda')
        # rotation_matrix = torch.tensor(
        #     Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_matrix(),
        #     dtype=torch.float32
        # ).to(pose_R.device)

        #pose_R = torch.matmul(rotation_matrix, pose_R)
        
        # pose_R = torch.tensor([
        #     [[1.0000,  0.0000,  0.0000],
        #     [ 0.0000, 1.0000,  0.0000],
        #     [ 0.0000,  0.0000,  1.0000]],
        #     Rotation.from_euler('xyz', [45, 0, 0], degrees=True).as_matrix(),
        #     Rotation.from_euler('xyz', [0, 45, 0], degrees=True).as_matrix(),
        #     Rotation.from_euler('xyz', [0, 0, 45], degrees=True).as_matrix(),
        #     Rotation.from_euler('zyx', [90, 45, 45], degrees=True).as_matrix(),
        # ], dtype=torch.float32).to('cuda')
        
        
        N_poses = pose_R.shape[0]
        select_pc = self.ref_data["pointcloud"][0].unsqueeze(0).repeat(N_poses, 1, 1)
        (N_query, N_pointcloud, _) = select_pc.shape
        posed_pc = torch.matmul(pose_R, select_pc.permute(0, 2, 1)).permute(0, 2, 1)
        translate = torch.tensor([ 0, 0,  0.4554]).repeat(N_poses, 1).to('cuda')
        posed_pc = posed_pc + translate[:, None, :].repeat(1, N_pointcloud, 1)
        cam_intrinsic = torch.tensor([[
            [1.5450e+03, 0.0000e+00, 9.8007e+02],
            [0.0000e+00, 1.5446e+03, 5.1264e+02],
            [0.0000e+00, 0.0000e+00, 1.0000e+00]]],
        dtype=torch.float64).to('cuda')
        cam_instrinsic = cam_intrinsic[0][None, ...].repeat(N_query, 1, 1).to(torch.float32)
        image_homo = torch.bmm(cam_instrinsic, posed_pc.permute(0, 2, 1)).permute(0, 2, 1)
        image_vu = (image_homo / image_homo[:, :, -1][:, :, None])[:, :, 0:2].to(torch.int) # N_query x N_pointcloud x 2
        (imageH, imageW) = (1080, 1920)
        image_vu[:, :, 0].clamp_(min=0, max=imageW - 1)
        image_vu[:, :, 1].clamp_(min=0, max=imageH - 1)
        
        for i in range(len(image_vu)):
            vis_pc = np.zeros((1080, 1920, 3), dtype=np.uint8)
            image_uv1 = image_vu[i]
            for point in image_uv1:
                x, y = int(point[0]), int(point[1])
                cv2.circle(vis_pc, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
            cv2.imwrite(f"debug/vis_template_pose/template_{i}.png", vis_pc)
        
    def project_template_to_image(self, best_pose, pred_object_idx, batch, proposals):
        """
        Obtain the RT of the best template, then project the reference pointclouds to query image, 
        getting the bbox of projected pointcloud from the image.
        """

        pose_R = self.ref_data["poses"][best_pose, 0:3, 0:3] # N_query x 3 x 3
        select_pc = self.ref_data["pointcloud"][pred_object_idx, ...] # N_query x N_pointcloud x 3
        (N_query, N_pointcloud, _) = select_pc.shape

        # translate object_selected pointcloud by the selected best pose and camera coordinate
        posed_pc = torch.matmul(pose_R, select_pc.permute(0, 2, 1)).permute(0, 2, 1)
        translate = self.Calculate_the_query_translation(proposals, batch["depth"][0], batch["cam_intrinsic"][0], batch['depth_scale'])
        posed_pc = posed_pc + translate[:, None, :].repeat(1, N_pointcloud, 1)
        # project the pointcloud to the image
        cam_instrinsic = batch["cam_intrinsic"][0][None, ...].repeat(N_query, 1, 1).to(torch.float32)
        image_homo = torch.bmm(cam_instrinsic, posed_pc.permute(0, 2, 1)).permute(0, 2, 1)
        image_vu = (image_homo / image_homo[:, :, -1][:, :, None])[:, :, 0:2].to(torch.int) # N_query x N_pointcloud x 2
        (imageH, imageW) = batch["depth"][0].shape
        image_vu[:, :, 0].clamp_(min=0, max=imageW - 1)
        image_vu[:, :, 1].clamp_(min=0, max=imageH - 1)

        return image_vu

    def Calculate_the_query_translation(self, proposal, depth, cam_intrinsic, depth_scale):
        """
        Calculate the translation amount from the origin of the object coordinate system to the camera coordinate system. 
        Cut out the depth using the provided mask and calculate the mean as the translation.
        proposal: N_query x imageH x imageW
        depth: imageH x imageW
        """
        (N_query, imageH, imageW) = proposal.squeeze_().shape
        masked_depth = proposal * (depth[None, ...].repeat(N_query, 1, 1))
        translate = depth_image_to_pointcloud_translate_torch(
            masked_depth, depth_scale, cam_intrinsic
        )
        return translate.to(torch.float32)

    def move_to_device(self):
        self.descriptor_model.model = self.descriptor_model.model.to(self.device)
        self.descriptor_model.model.device = self.device
        # if there is predictor in the model, move it to device
        if hasattr(self.segmentor_model, "predictor"):
            self.segmentor_model.predictor.model = (
                self.segmentor_model.predictor.model.to(self.device)
            )
        else:
            self.segmentor_model.model.setup_model(device=self.device, verbose=True)
        logging.info(f"Moving models to {self.device} done!")

    def compute_semantic_score(self, proposal_descriptors):
        torch.cuda.synchronize()
        semantic_score_start_time = time.time()
        
        start_time = time.time()
        # compute matching scores for each proposals
        scores = self.matching_config.metric(
            proposal_descriptors, self.ref_data["descriptors"]
        )  # N_proposals x N_objects x N_templates
        torch.cuda.synchronize()
        print(f"Compute idx selected proposals (part 1): {time.time()-start_time}s")
        
        start_time = time.time()
        if self.matching_config.aggregation_function == "mean":
            score_per_proposal_and_object = (
                torch.sum(scores, dim=-1) / scores.shape[-1]
            )  # N_proposals x N_objects
        elif self.matching_config.aggregation_function == "median":
            score_per_proposal_and_object = torch.median(scores, dim=-1)[0]
        elif self.matching_config.aggregation_function == "max":
            score_per_proposal_and_object = torch.max(scores, dim=-1)[0]
        elif self.matching_config.aggregation_function == "avg_5":
            score_per_proposal_and_object = torch.topk(scores, k=5, dim=-1)[0]
            score_per_proposal_and_object = torch.mean(
                score_per_proposal_and_object, dim=-1
            )
        else:
            raise NotImplementedError
        torch.cuda.synchronize()
        print(f"Compute idx selected proposals (part 1.5): {time.time()-start_time}s")
        start_time = time.time()
        # assign each proposal to the object with highest scores
        score_per_proposal, assigned_idx_object = torch.max(
            score_per_proposal_and_object, dim=-1
        )  # N_query
        torch.cuda.synchronize()
        print(f"Compute idx selected proposals (part 1.6): {time.time()-start_time}s")
        
        start_time = time.time()
        criteria = score_per_proposal > self.matching_config.confidence_thresh
        
        proposal_arange = torch.arange(
            score_per_proposal.shape[0], device=score_per_proposal.device
        )
        torch.cuda.synchronize()
        print(f"Compute idx selected proposals (part 1.9): {time.time()-start_time}s")
        
        start_time = time.time()
        idx_selected_proposals = proposal_arange[criteria]
        #idx_selected_proposals = proposal_arange[torch.nonzero(criteria, as_tuple=True)]
        torch.cuda.synchronize()
        print(f"Compute idx selected proposals (part 2): {time.time()-start_time}s")
        
        
        start_time = time.time()
        pred_idx_objects = assigned_idx_object[idx_selected_proposals]
        semantic_score = score_per_proposal[idx_selected_proposals]
        torch.cuda.synchronize()
        print(f"Compute idx selected proposals (part 3): {time.time()-start_time}s")
        
        start_time = time.time()
        # compute the best view of template
        flitered_scores = scores[idx_selected_proposals, ...]
        best_template = self.best_template_pose(flitered_scores, pred_idx_objects)
        torch.cuda.synchronize()
        print(f"Compute idx selected proposals (part 4): {time.time()-start_time}s")
        
        print(f"Compute semantic score entire function: {time.time()-semantic_score_start_time}s")
        return idx_selected_proposals, pred_idx_objects, semantic_score, best_template

    def compute_appearance_score(self, best_pose, pred_objects_idx, query_appe_descriptors):
        """
        Based on the best template, calculate appearance similarity indicated by appearance score
        """
        con_idx = torch.concatenate((pred_objects_idx[None, :], best_pose[None, :]), dim=0)
        ref_appe_descriptors = self.ref_data["appe_descriptors"][con_idx[0, ...], con_idx[1, ...], ...] # N_query x N_patch x N_feature

        aux_metric = MaskedPatch_MatrixSimilarity(metric="cosine", chunk_size=64)
        appe_scores = aux_metric.compute_straight(query_appe_descriptors, ref_appe_descriptors)

        return appe_scores, ref_appe_descriptors

    def estimate_orientation(self, pred_objects_idx, query_appe_descriptor):
        """
        Based on the best template, calculate appearance similarity indicated by appearance score
        """
        ref_appe_descriptors = self.ref_data["appe_descriptors"][pred_objects_idx]
        similarity_scores = torch.nn.functional.cosine_similarity(
            ref_appe_descriptors, query_appe_descriptor.unsqueeze(0), dim=-1
        )
        mean_similarity_scores = similarity_scores.mean(dim=-1)
        best_template_idx = torch.argmax(mean_similarity_scores)
        return int(best_template_idx)
        
    def compute_geometric_score(self, image_uv, proposals, appe_descriptors, ref_aux_descriptor, visible_thred=0.5):

        aux_metric = MaskedPatch_MatrixSimilarity(metric="cosine", chunk_size=64)
        visible_ratio = aux_metric.compute_visible_ratio(appe_descriptors, ref_aux_descriptor, visible_thred)
        
        # IoU calculation
        y1x1 = torch.min(image_uv, dim=1).values
        y2x2 = torch.max(image_uv, dim=1).values
        xyxy = torch.concatenate((y1x1, y2x2), dim=-1)
        
        iou = compute_iou(xyxy, proposals.boxes)

        return iou, visible_ratio

    def compute_visible_ratio(self, appe_descriptors, ref_aux_descriptor, visible_thred=0.5):
        aux_metric = MaskedPatch_MatrixSimilarity(metric="cosine", chunk_size=64)
        visible_ratio = aux_metric.compute_visible_ratio(appe_descriptors, ref_aux_descriptor, visible_thred)
        return visible_ratio

    def test_step(self, batch, idx):
        if idx == 0:
            os.makedirs(
                osp.join(
                    self.log_dir,
                    f"predictions/{self.dataset_name}/{self.name_prediction_file}",
                ),
                exist_ok=True,
            )
            self.set_reference_objects()
            self.set_reference_object_pointcloud()
            self.move_to_device()
        assert batch["image"].shape[0] == 1, "Batch size must be 1"

        image_np = (
            self.inv_rgb_transform(batch["image"][0])
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )
        image_np = np.uint8(image_np.clip(0, 1) * 255)

        # run propoals
        proposal_stage_start_time = time.time()
        proposals = self.segmentor_model.generate_masks(image_np)

        # init detections with masks and boxes
        detections = Detections(proposals)
        detections.remove_very_small_detections(
            config=self.post_processing_config.mask_post_processing
        )

        # compute semantic descriptors and appearance descriptors for query proposals
        query_decriptors, query_appe_descriptors = self.descriptor_model(image_np, detections)
        proposal_stage_end_time = time.time()

        # matching descriptors
        matching_stage_start_time = time.time()
        (
            idx_selected_proposals,
            pred_idx_objects,
            semantic_score,
            best_template,
        ) = self.compute_semantic_score(query_decriptors)

        # update detections
        detections.filter(idx_selected_proposals)
        query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

        # compute the appearance score
        appe_scores, ref_aux_descriptor= self.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

        # compute the geometric score
        image_uv = self.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

        geometric_score, visible_ratio = self.compute_geometric_score(
            image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=self.visible_thred
            )

        # final score
        final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)

        detections.add_attribute("scores", final_score)
        detections.add_attribute("object_ids", pred_idx_objects)
        detections.apply_nms_per_object_id(
            nms_thresh=self.post_processing_config.nms_thresh
        )
        matching_stage_end_time = time.time()

        runtime = (
            proposal_stage_end_time
            - proposal_stage_start_time
            + matching_stage_end_time
            - matching_stage_start_time
        )
        detections.to_numpy()

        scene_id = batch["scene_id"][0]
        frame_id = batch["frame_id"][0]
        file_path = osp.join(
            self.log_dir,
            f"predictions/{self.dataset_name}/{self.name_prediction_file}/scene{scene_id}_frame{frame_id}",
        )

        # save detections to file
        results = detections.save_to_file(
            scene_id=int(scene_id),
            frame_id=int(frame_id),
            runtime=runtime,
            file_path=file_path,
            dataset_name=self.dataset_name,
            return_results=True,
        )
        # save runtime to file
        np.savez(
            file_path + "_runtime",
            proposal_stage=proposal_stage_end_time - proposal_stage_start_time,
            matching_stage=matching_stage_end_time - matching_stage_start_time,
        )
        return 0

    def test_epoch_end(self, outputs):
        if self.global_rank == 0:  # only rank 0 process
            # can use self.all_gather to gather results from all processes
            # but it is simpler just load the results from files so no file is missing
            result_paths = sorted(
                glob.glob(
                    osp.join(
                        self.log_dir,
                        f"predictions/{self.dataset_name}/{self.name_prediction_file}/*.npz",
                    )
                )
            )
            result_paths = sorted(
                [path for path in result_paths if "runtime" not in path]
            )
            num_workers = 10
            logging.info(f"Converting npz to json requires {num_workers} workers ...")
            pool = multiprocessing.Pool(processes=num_workers)
            convert_npz_to_json_with_idx = partial(
                convert_npz_to_json,
                list_npz_paths=result_paths,
            )
            detections = list(
                tqdm(
                    pool.imap_unordered(
                        convert_npz_to_json_with_idx, range(len(result_paths))
                    ),
                    total=len(result_paths),
                    desc="Converting npz to json",
                )
            )
            formatted_detections = []
            for detection in tqdm(detections, desc="Loading results ..."):
                formatted_detections.extend(detection)

            detections_path = f"{self.log_dir}/{self.name_prediction_file}.json"
            save_json_bop23(detections_path, formatted_detections)
            logging.info(f"Saved predictions to {detections_path}")
