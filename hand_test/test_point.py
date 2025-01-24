from pathlib import Path
import torch
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import cv2
import numpy as np
import pickle

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.utils.geometry import aa_to_rotmat, perspective_projection
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer.utils.render_openpose import render_openpose
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from typing import List, Tuple
openpose_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
gt_indices = openpose_indices


def convert_crop_coords_to_orig_img(bbox, keypoints, crop_size):
    # import IPython; IPython.embed(); exit()
    cx, cy, h = bbox[:, 0], bbox[:, 1], bbox[:, 2]

    # unnormalize to crop coords
    # keypoints = 0.5 * crop_size * (keypoints + 1.0)

    # rescale to orig img crop
    keypoints *= h[..., None, None] / crop_size

    # transform into original image coords
    keypoints[:,:,0] = (cx - h/2)[..., None] + keypoints[:,:,0]
    keypoints[:,:,1] = (cy - h/2)[..., None] + keypoints[:,:,1]
    return keypoints









def save_video(path, out_name):
    print('saving to :', out_name + '.mp4')
    img_array = []
    height, width = 0, 0
    for filename in tqdm(sorted(os.listdir(path), key=lambda x:int(x.split('.')[0]))):
        img = cv2.imread(path + '/' + filename)
        if height != 0:
            img = cv2.resize(img, (width, height))
        height, width, _ = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter(out_name + '.mp4', 0x7634706d, 30, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print('done')

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--res_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')

    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hamer
    cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    args.out_folder = args.out_folder + '_' +str(model_cfg.EXTRA.FOCAL_LENGTH)
    os.makedirs(args.out_folder, exist_ok=True)
    render_save_path = os.path.join(args.out_folder, 'render_all')
    depth_save_path = os.path.join(args.out_folder, 'depth_all')
    joint2d_save_path = os.path.join(args.out_folder, 'joint2d')
    vit_save_path = os.path.join(args.out_folder, 'vit')
    os.makedirs(render_save_path, exist_ok=True)
    os.makedirs(joint2d_save_path, exist_ok=True)
    os.makedirs(vit_save_path, exist_ok=True)
    os.makedirs(depth_save_path, exist_ok=True)

    # Get all demo images ends with .jpg or .png
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]

    # Iterate over all images in folder
    img_list = []
    print('start iteration.')
    results_dict = {}

    for img_path in tqdm(sorted(img_paths)):
        img_path = str(img_path)
        img_cv2 = cv2.imread(str(img_path))
        results_dict[img_path] = {}
        results_dict[img_path]['mano'] = []
        results_dict[img_path]['cam_trans'] = []
        results_dict[img_path]['extra_data'] = []
        # if '000009' not in str(img_path) and '000008' not in str(img_path):
        #     continue

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()
        # print(det_out.keys())
        # print(';******************')
        # print(det_instances)

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img_cv2,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []
        vit_keypoints_list = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            # print(vitposes.keys()) # dict_keys(['bbox', 'keypoints'])
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 10:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(0)
                vit_keypoints_list.append(keyp)

            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 10:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(1)
                vit_keypoints_list.append(keyp)

        if len(bboxes) == 0:
            results_dict[img_path]['tid'] = tid
            results_dict[img_path]['tracked_time'] = []
            results_dict[img_path]['shot'] = 0
            tracked_time[0] += 1
            tracked_time[1] += 1
            for i in tid:
                results_dict[img_path]['tracked_time'].append(tracked_time[i])
            for idx, i in enumerate(tracked_time):
                if i > 50 and (idx in tid):
                    tid.remove(idx)
            print('no hand detected!!!', results_dict[img_path]['tid'], results_dict[img_path]['tracked_ids'], results_dict[img_path]['tracked_time'])
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        vit_keypoints = np.stack(vit_keypoints_list)
        # if len(left_hand_keyp_list) > 0:
        #     l_vit_keypoints = np.stack(left_hand_keyp_list)
        # else:
        #     l_vit_keypoints = np.zeros((1, 21, 3))
        # if len(right_hand_keyp_list) > 0:
        #     r_vit_keypoints = np.stack(right_hand_keyp_list)
        # else:
        #     r_vit_keypoints = np.zeros((1, 21, 3))
        # print('raw vit results:', l_vit_keypoints.shape, r_vit_keypoints.shape)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right,  rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        all_vit_2d = []
        all_pred_2d = []
        all_bboxes = []

        left_flag = False
        right_flag = False
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            # print(scaled_focal_length, model_cfg.EXTRA.FOCAL_LENGTH, model_cfg.MODEL.IMAGE_SIZE, img_size.max())
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)#.detach().cpu().numpy()

            # Render the result，before passing to project
            batch_size = batch['img'].shape[0]
            pred_keypoints_3d = out['pred_keypoints_3d'].reshape(batch_size, -1, 3)
            # is_right = int(batch['right'].cpu().numpy())
            pred_keypoints_3d[:,:,0] = multiplier*pred_keypoints_3d[:,:,0]
            out['pred_keypoints_2d'] = perspective_projection(pred_keypoints_3d,
                                        translation=pred_cam_t_full.reshape(batch_size, 3),
                                        focal_length=torch.tensor([[scaled_focal_length, scaled_focal_length]]),
                                        camera_center=torch.tensor([704,704]))# out['focal_length'].reshape(-1, 2) / model_cfg.MODEL.IMAGE_SIZE)

            pred_cam_t_full = pred_cam_t_full.detach().cpu().numpy()

            # print('batch_size: ', batch_size)
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])
                white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()

                # regression_img, regression_depth = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                #                         out['pred_cam_t'][n].detach().cpu().numpy(),
                #                         batch['img'][n],
                #                         mesh_base_color=LIGHT_BLUE,
                #                         scene_bg_color=(1, 1, 1),
                #                         )

                # if args.side_view:
                #     side_img, side_depth = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                #                             out['pred_cam_t'][n].detach().cpu().numpy(),
                #                             white_img,
                #                             mesh_base_color=LIGHT_BLUE,
                #                             scene_bg_color=(1, 1, 1),
                #                             side_view=True)
                #     final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                # else:
                #     final_img = np.concatenate([input_patch, regression_img], axis=1)

                # cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                pred_joints = out['pred_keypoints_2d'][n].detach().cpu().numpy()

                is_right = int(batch['right'][n].cpu().numpy())
                #这里不用转？
                # pred_joints[:,0] = (2*is_right-1)*pred_joints[:,0]
                v = np.ones((21, 1))
                pred_joints = np.concatenate((pred_joints, v), axis=-1)
                verts[:,0] = (2*is_right-1)*verts[:,0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)
                all_pred_2d.append(pred_joints)
                # all_vit_2d.append(batch['2d'][n])
                # all_bboxes.append(batch['bbox'][n].detach().cpu().numpy())

                # results_dict[img_path]['mano'].append(out['pred_mano_params'][n])
                # results_dict[img_path]['cam_trans'].append(cam_t)

                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{is_right}.obj'))

        # Render front view
        # assert len(all_vit_2d) == len(all_pred_2d)
        # print('length: ', len(all_vit_2d), len(all_pred_2d))
        # all_vit_2d = torch.stack(all_vit_2d).cpu().numpy()
        all_pred_2d = np.stack(all_pred_2d)
        # all_bboxes = np.stack(all_bboxes)
        # print('vit_2d: ', all_vit_2d.shape)
        print('pred_2d: ', all_pred_2d.shape)
        # print('all_bboxes: ', all_bboxes.shape)

        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            # print(all_cam_t)
            cam_view= renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            # vit_img = input_img.copy()[:,:,::-1] * 255
            # for i in range(len(all_verts)):
            #     body_keypoints_2d = all_vit_2d[i, :21].copy()
            #     for op, gt in zip(openpose_indices, gt_indices):
            #         if all_vit_2d[i, gt, -1] > body_keypoints_2d[op, -1]:
            #             body_keypoints_2d[op] = all_vit_2d[i, gt]
            #     vit_img = render_openpose(vit_img, body_keypoints_2d)

            pred_img = input_img.copy()[:,:,:-1][:,:,::-1] * 255
            print(all_pred_2d, pred_img.shape)
            # exit()
            # all_pred_2d = model_cfg.MODEL.IMAGE_SIZE * (all_pred_2d + 1) * 0.5
            # all_pred_2d = convert_crop_coords_to_orig_img(bbox=all_bboxes, keypoints=all_pred_2d, crop_size=model_cfg.MODEL.IMAGE_SIZE)
            for i in range(len(all_verts)):
                body_keypoints_2d = all_pred_2d[i, :21].copy()
                for op, gt in zip(openpose_indices, gt_indices):
                    if all_pred_2d[i, gt, -1] > body_keypoints_2d[op, -1]:
                        body_keypoints_2d[op] = all_pred_2d[i, gt]
                pred_img = render_openpose(pred_img, body_keypoints_2d)

            # draw 2d keypoints
            cv2.imwrite(os.path.join(render_save_path, f'{img_fn}.jpg'), 255*input_img_overlay[:, :, ::-1])
            cv2.imwrite(os.path.join(joint2d_save_path, f'{img_fn}.jpg'), pred_img[:, :, ::-1])
            # cv2.imwrite(os.path.join(vit_save_path, f'{img_fn}.jpg'), vit_img[:, :, ::-1])

    save_video(render_save_path, args.out_folder)
    save_video(joint2d_save_path, args.out_folder + '_2d')
    save_video(vit_save_path, args.out_folder + '_vit')

if __name__ == '__main__':
    main()
