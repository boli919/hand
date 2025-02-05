import cv2
import numpy as np
import json
from typing import Optional, List, Tuple
import torch
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
from pathlib import Path
import rerun as rr
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from hamer.utils.geometry import aa_to_rotmat, perspective_projection
from projectaria_tools.core import calibration, data_provider, mps, image
from projectaria_tools.core.calibration import CameraCalibration, DeviceCalibration
from projectaria_tools.core.mps import MpsDataPathsProvider
from projectaria_tools.core.mps.utils import (
    get_nearest_pose,
    get_nearest_wrist_and_palm_pose,
)
from projectaria_tools.core.sensor_data import SensorData, SensorDataType, TimeDomain
from projectaria_tools.core.stream_id import StreamId

from vitpose_model import ViTPoseModel
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
import hamer
from hamer.utils import recursive_to
from detectron2.config import LazyConfig
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer.utils.render_openpose import render_hand_keypoints, render_openpose
# import hamer.utils.render_openpose 
import cProfile

openpose_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
gt_indices = openpose_indices
# 以下为手部关键点检测和绘制需要的依赖
WRIST_PALM_TIME_DIFFERENCE_THRESHOLD_NS: int = 2e8
WRIST_PALM_COLOR: List[int] = [255, 64, 0]
NORMAL_VIS_LEN_2D = 30.0  # in pixels

from pathlib import Path
from typing import List, Tuple

def find_vrs_and_handtracking(base_folder: str) -> List[Tuple[Path, Path]]:
    """
    在给定的基文件夹中自动匹配 .vrs 文件和 hand_tracking 数据文件。
    
    Args:
        base_folder (str): 数据集顶层文件夹路径
    
    Returns:
        List[Tuple[Path, Path]]: 每个匹配项为 (vrs_file, handtracking_file)
    """
    base_folder = Path(base_folder)
    experiments = [d for d in base_folder.iterdir() if d.is_dir() and not d.name.startswith('.')]
    matched_files = []

    for exp in experiments:
        # 找到 ego 文件夹
        ego_folder = exp / "ego"
        if not ego_folder.exists():
            print(f"No 'ego' folder found in {exp}")
            continue

        # 找到 ego 文件夹中的 .vrs 文件
        vrs_file = next(ego_folder.glob("*.vrs"), None)
        if not vrs_file:
            print(f"No .vrs file found in {ego_folder}")
            continue

        # 找到 mps_*_vrs 文件夹
        mps_folder = next(ego_folder.glob("mps_*_vrs"), None)
        if not mps_folder or not mps_folder.is_dir():
            print(f"No mps folder found in {ego_folder}")
            continue

        # 找到 hand_tracking 文件夹
        handtracking_folder = mps_folder / "hand_tracking"
        if not handtracking_folder.exists():
            print(f"No hand_tracking folder found in {mps_folder}")
            continue

        # 找到 hand_tracking 文件夹中的 .csv 文件
        handtracking_file = next(handtracking_folder.glob("*.csv"), None)
        if not handtracking_file:
            print(f"No hand_tracking CSV file found in {handtracking_folder}")
            continue

        # 匹配成功，加入列表
        matched_files.append((vrs_file, handtracking_file))

    return matched_files

def get_camera_projection_from_device_point(
    point: np.ndarray, camera_calibration
) -> Optional[np.ndarray]:
    # print(f"received point:{point}")
    T_device_camera = camera_calibration.get_transform_device_camera()
    # print(camera_calibration)
    # print(f"required type:{type(T_device_camera.inverse() @ point)}")
    return camera_calibration.project(T_device_camera.inverse() @ point)


def log_hand_tracking(
    wrist_and_palm_poses,
    device_time_ns: int,
    rgb_camera_calibration,
    rgb_stream_label: str,
    down_sampling_factor: int,
    closed_loop_traj,
    future_interval_seconds: float = 1.0,
    num_samples: int = 10,
) -> str:
    # print(f"left and right:{wrist_and_palm_poses.keys()}")
    if not wrist_and_palm_poses:
        return json.dumps({"left_hand": [], "right_hand": []}, indent=2)

    wrist_and_palm_pose = get_nearest_wrist_and_palm_pose(wrist_and_palm_poses, device_time_ns)
    # print(f"keys:{wrist_and_palm_pose.keys()}")
    if wrist_and_palm_pose is None:
        return json.dumps({"left_hand": [], "right_hand": []}, indent=2)

    left_hand_points = []
    right_hand_points = []

    # 获取当前帧的设备到世界的变换矩阵
    current_pose_info = get_nearest_pose(closed_loop_traj, device_time_ns)
    if current_pose_info is None:
        return json.dumps({"left_hand": [], "right_hand": []}, indent=2)
    T_world_device_current = current_pose_info.transform_world_device

    # Process left hand
    # print(f"left:{wrist_and_palm_pose.left_hand}")
    if wrist_and_palm_pose.left_hand and wrist_and_palm_pose.left_hand.confidence >= 0:
        left_wrist_point_device = wrist_and_palm_pose.left_hand.wrist_position_device
        left_wrist_pixel = get_camera_projection_from_device_point(left_wrist_point_device, rgb_camera_calibration)
        # print(f"left:{left_wrist_pixel}")

        if left_wrist_pixel is not None:
            scaled_point = [p / down_sampling_factor for p in left_wrist_pixel]
            left_hand_points.append(scaled_point)
            
    # Process right hand
    if wrist_and_palm_pose.right_hand and wrist_and_palm_pose.right_hand.confidence > 0:
        right_wrist_point_device = wrist_and_palm_pose.right_hand.wrist_position_device
        right_wrist_pixel = get_camera_projection_from_device_point(right_wrist_point_device, rgb_camera_calibration)

        if right_wrist_pixel is not None:
            scaled_point = [p / down_sampling_factor for p in right_wrist_pixel]
            right_hand_points.append(scaled_point)

    # Predict future points for the next 1 second
    future_interval_ns = int(future_interval_seconds * 1e9)
    t_start = device_time_ns
    t_end = device_time_ns + future_interval_ns
    future_timestamps = np.linspace(t_start, t_end, num_samples)

    for t_future in future_timestamps:
        future_wrist_and_palm_pose = get_nearest_wrist_and_palm_pose(wrist_and_palm_poses, t_future)
        if future_wrist_and_palm_pose is None:
            continue

        # 获取未来帧的设备到世界的变换矩阵
        future_pose_info = get_nearest_pose(closed_loop_traj, t_future)
        if future_pose_info is None:
            continue
        T_world_device_future = future_pose_info.transform_world_device

        # Process future left hand points
        if future_wrist_and_palm_pose.left_hand and future_wrist_and_palm_pose.left_hand.confidence > 0:
            future_left_wrist_point_device = future_wrist_and_palm_pose.left_hand.wrist_position_device
            future_left_wrist_point_world = T_world_device_future @ future_left_wrist_point_device
            future_left_wrist_point_device_first = T_world_device_current.inverse() @ future_left_wrist_point_world
            future_left_wrist_pixel = get_camera_projection_from_device_point(future_left_wrist_point_device_first, rgb_camera_calibration)

            if future_left_wrist_pixel is not None:
                future_scaled_point = [p / down_sampling_factor for p in future_left_wrist_pixel]
                left_hand_points.append(future_scaled_point)

        # Process future right hand points
        if future_wrist_and_palm_pose.right_hand and future_wrist_and_palm_pose.right_hand.confidence > 0:
            future_right_wrist_point_device = future_wrist_and_palm_pose.right_hand.wrist_position_device
            future_right_wrist_point_world = T_world_device_future @ future_right_wrist_point_device
            future_right_wrist_point_device_first = T_world_device_current.inverse() @ future_right_wrist_point_world
            future_right_wrist_pixel = get_camera_projection_from_device_point(future_right_wrist_point_device_first, rgb_camera_calibration)

            if future_right_wrist_pixel is not None:
                future_scaled_point = [p / down_sampling_factor for p in future_right_wrist_pixel]
                right_hand_points.append(future_scaled_point)

    result = {
        "left_hand": [
            {"wrist_point": point} for point in left_hand_points
        ] if len(left_hand_points) >= 2 else [],
        "right_hand": [
            {"wrist_point": point} for point in right_hand_points
        ] if len(right_hand_points) >= 2 else [],
    }
    # print(f"result:{result}")
    result_json = json.dumps(result, indent=2)
    # print(f"hand projections:{result_json}")
    return result_json

def adjust_white_balance(image):
    result = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    avg_a = np.mean(result[:, :, 1])
    avg_b = np.mean(result[:, :, 2])
    result[:, :, 1] = np.clip(result[:, :, 1] - (avg_a - 128), 0, 255).astype(np.uint8)
    result[:, :, 2] = np.clip(result[:, :, 2] - (avg_b - 128), 0, 255).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
    return result

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vrs", type=str, help="path to VRS file")
    parser.add_argument("--trajectory", nargs="+", type=str, help="path(s) to MPS closed-loop trajectory files")
    parser.add_argument("--points", nargs="+", type=str, help="path(s) to the MPS global points file")
    parser.add_argument("--eyegaze", type=str, help="path to the MPS eye gaze file")
    parser.add_argument("--hands", type=str, help="path to the MPS hand tracking file")
    parser.add_argument("--mps_folder", type=str, help="path to the MPS folder (will overwrite default value <vrs_file>/mps)")
    #no compression
    parser.add_argument("--down_sampling_factor", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--jpeg_quality", type=int, default=75, help=argparse.SUPPRESS)
    parser.add_argument("--rrd_output_path", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--no_rotate_image_upright", action="store_true", help="If set, the RGB images are shown in their original orientation.")
    parser.add_argument("--no_rectify_image", action="store_true", help="If set, the raw fisheye RGB images are shown without being undistorted.")
    return parser.parse_args()


# def draw_keypoints_and_connections(image, keypoints, confidence_threshold=0.8):
#     """
#     在图像上绘制关键点并连接关键点
#     """
#     # 手部关键点连接顺序（21个关键点，典型手部COCO格式）
#     connections = [
#         [0, 1, 2, 3, 4],  # 拇指
#         [0, 5, 6, 7, 8],  # 食指
#         [0, 9, 10, 11, 12],  # 中指
#         [0, 13, 14, 15, 16],  # 无名指
#         [0, 17, 18, 19, 20]  # 小指
#     ]

#     # image_2=iamge.copy()
#     for conn in connections:
#         for i in range(len(conn) - 1):
#             kp1, kp2 = keypoints[conn[i]], keypoints[conn[i + 1]]
#             if kp1[2] > confidence_threshold and kp2[2] > confidence_threshold:
#                 x1, y1 = int(kp1[0]), int(kp1[1])
#                 x2, y2 = int(kp2[0]), int(kp2[1])
#                 cv2.line(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)
#         # for kp in keypoints:
#         # if kp[2] > confidence_threshold:
#         #     x, y = int(kp[0]), int(kp[1])
#         #     cv2.circle(image, (x, y), radius=1, color=(0, 255, 0), thickness=-1)
#     #no hamer
#     # for conn in connections:
#     #     for i in range(len(conn) - 1):
#     #         kp1, kp2 = keypoints[conn[i]], keypoints[conn[i + 1]]
#     #         if kp1[2] > confidence_threshold and kp2[2] > confidence_threshold:
#     #             x1, y1 = int(kp1[0]), int(kp1[1])
#     #             x2, y2 = int(kp2[0]), int(kp2[1])
#     #             cv2.line(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)


#     return image

def detect_and_draw_hands(
    img: np.ndarray, 
    model: HAMER, 
    camera_calibration,
    model_cfg,
    detector,
    cpm,
    hand_projections: Optional[str] = None,
    renderer=None,
    device=None,
) -> np.ndarray:   
    """
    使用HAMER检测手部关键点并绘制，使用实际手腕点替换检测的手腕点
    """
    img_copy = img.copy()
    left_wrist_point = None
    right_wrist_point = None
    
    # 获取实际手腕点坐标
    if hand_projections:
        try:
            hand_data = json.loads(hand_projections)
            
            # 获取左手最新的手腕点
            left_hand_points = hand_data.get("left_hand", [])
            if left_hand_points:
                left_wrist_point = left_hand_points[0]["wrist_point"]
                for point in left_hand_points:
                    wrist_point = point["wrist_point"]
                    x, y = wrist_point[0], wrist_point[1]
                    cv2.circle(img_copy, (int(x), int(y)), 1, (255, 255, 0), -1)
                
                for i in range(1, len(left_hand_points)):
                    curr_point = left_hand_points[i]["wrist_point"]
                    prev_point = left_hand_points[i-1]["wrist_point"]
                    cv2.line(img_copy, 
                            (int(prev_point[0]), int(prev_point[1])),
                            (int(curr_point[0]), int(curr_point[1])),
                            (255, 255, 0), 3)
            
            # 获取右手最新的手腕点
            right_hand_points = hand_data.get("right_hand", [])
            if right_hand_points:
                right_wrist_point = right_hand_points[0]["wrist_point"]
                for point in right_hand_points:
                    wrist_point = point["wrist_point"]
                    x, y = wrist_point[0], wrist_point[1]
                    cv2.circle(img_copy, (int(x), int(y)), 1, (255, 255, 0), -1)
                
                for i in range(1, len(right_hand_points)):
                    curr_point = right_hand_points[i]["wrist_point"]
                    prev_point = right_hand_points[i-1]["wrist_point"]
                    cv2.line(img_copy, 
                            (int(prev_point[0]), int(prev_point[1])),
                            (int(curr_point[0]), int(curr_point[1])),
                            (255, 255, 0), 3)
                            
        except Exception as e:
            print(f"Error processing hand projections: {e}")
    
    img_cv2 = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
    
    det_out = detector(img_cv2)
    img = img_cv2.copy()[:, :, ::-1]

    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()

    vitposes_out = cpm.predict_pose(
        img_cv2,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )

    bboxes = []
    is_right = []

    for vitposes in vitposes_out:
        # Initialize variables for both hands
        left_hand_bbox = None
        right_hand_bbox = None
        left_max_conf = 0
        right_max_conf = 0
        
        # Process left hand keypoints
        left_hand_keyp = vitposes['keypoints'][-42:-21]
        valid_left = left_hand_keyp[:,2] > 0.5
        if sum(valid_left) > 3:
            conf = np.mean(left_hand_keyp[valid_left, 2])
            if conf > left_max_conf:
                left_max_conf = conf
                left_hand_bbox = [
                    left_hand_keyp[valid_left,0].min(),
                    left_hand_keyp[valid_left,1].min(),
                    left_hand_keyp[valid_left,0].max(),
                    left_hand_keyp[valid_left,1].max()
                ]
        
        # Process right hand keypoints
        right_hand_keyp = vitposes['keypoints'][-21:]
        valid_right = right_hand_keyp[:,2] > 0.5
        if sum(valid_right) > 3:
            conf = np.mean(right_hand_keyp[valid_right, 2])
            if conf > right_max_conf:
                right_max_conf = conf
                right_hand_bbox = [
                    right_hand_keyp[valid_right,0].min(),
                    right_hand_keyp[valid_right,1].min(),
                    right_hand_keyp[valid_right,0].max(),
                    right_hand_keyp[valid_right,1].max()
                ]

        # Add valid bboxes to the lists
        if left_hand_bbox is not None:
            bboxes.append(left_hand_bbox)
            is_right.append(0)
        if right_hand_bbox is not None:
            bboxes.append(right_hand_bbox)
            is_right.append(1)

    if len(bboxes) == 0:
        print("No valid bounding boxes detected. Returning original image.")
        return img_copy

    boxes = np.stack(bboxes)
    right = np.stack(is_right)
    
    dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=2.0)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )

    # Rest of your function remains the same...
    # [Previous code for processing predictions and drawing remains unchanged]

    all_verts = []
    all_cam_t = []
    all_right = []
    all_vit_2d = []
    all_pred_2d = []
    all_bboxes = []

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
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)

        batch_size = batch['img'].shape[0]
           
        pred_keypoints_3d = out['pred_keypoints_3d'].reshape(batch_size, -1, 3)
        for i in range(batch_size):
            current_multiplier = multiplier[i]
            pred_keypoints_3d[i,:,0] = current_multiplier * pred_keypoints_3d[i,:,0]

        out['pred_keypoints_2d'] = perspective_projection(pred_keypoints_3d,
                                    translation=pred_cam_t_full.reshape(batch_size, 3),
                                    focal_length=torch.tensor([[scaled_focal_length, scaled_focal_length]]),
                                    camera_center=torch.tensor([703.5,703.5]))
        
        pred_cam_t_full = pred_cam_t_full.detach().cpu().numpy()
        
        for n in range(batch_size):
            person_id = int(batch['personid'][n])
            white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
            input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
            input_patch = input_patch.permute(1,2,0).numpy()

            verts = out['pred_vertices'][n].detach().cpu().numpy()
            pred_joints = out['pred_keypoints_2d'][n].detach().cpu().numpy()
            is_right = int(batch['right'][n].cpu().numpy())
            
            # 根据是否为右手替换手腕点坐标
            if is_right and right_wrist_point is not None:
                print(f"right")
                pred_joints[0, 0] = right_wrist_point[0]  # x坐标
                pred_joints[0, 1] = right_wrist_point[1]  # y坐标
                
            elif not is_right and left_wrist_point is not None:
                print(f"left")
                pred_joints[0, 0] = left_wrist_point[0]  # x坐标
                pred_joints[0, 1] = left_wrist_point[1]  # y坐标
            
            v = np.ones((21, 1))
            pred_joints = np.concatenate((pred_joints, v), axis=-1)
            verts[:,0] = (2*is_right-1)*verts[:,0]
            cam_t = pred_cam_t_full[n]
            all_verts.append(verts)
            all_cam_t.append(cam_t)
            all_right.append(is_right)
            all_pred_2d.append(pred_joints)
            
    all_pred_2d = np.stack(all_pred_2d)

    # 添加判断条件，舍弃第一行数据差值不超过5的点
    to_remove = set()
    for i in range(len(all_pred_2d)):
        for j in range(i + 1, len(all_pred_2d)):
            if np.abs(all_pred_2d[i, 0, 0] - all_pred_2d[j, 0, 0]) <= 5:
                to_remove.add(j)

    all_pred_2d = np.delete(all_pred_2d, list(to_remove), axis=0)

    input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
    input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2)
    pred_img = input_img.copy()[:,:,:-1][:,:,::-1] * 255
    
    for i in range(len(all_pred_2d)):
        body_keypoints_2d = all_pred_2d[i, :21].copy()
        for op, gt in zip(openpose_indices, gt_indices):
            if all_pred_2d[i, gt, -1] > body_keypoints_2d[op, -1]:
                body_keypoints_2d[op] = all_pred_2d[i, gt]
        pred_img = render_openpose(pred_img, body_keypoints_2d)
    
    output_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
    
    return output_img

def log_RGB_image(
    data,
    down_sampling_factor: int,
    jpeg_quality: int,
    rgb_stream_label: str,
    output_base_dir: str,
    camera_calibration,
    session_index: int,
    hand_projections: Optional[str] = None,
    postprocess_image = lambda img: img,
    hamer_model: Optional[HAMER] = None,  # 替换detector和cpm参数
    model_cfg = None,
    detector = None,
    keypoint_detector = None,
    renderer=None,
    device=None,
):
    if data.sensor_data_type() == SensorDataType.IMAGE:
        img = data.image_data_and_record()[0].to_numpy_array()
        img = postprocess_image(img)
        if down_sampling_factor > 1:
            img = img[::down_sampling_factor, ::down_sampling_factor]
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)

        # 使用HAMER进行手部检测和绘制
        if hamer_model is not None:
            img= detect_and_draw_hands(
                img, 
                hamer_model, 
                camera_calibration,
                model_cfg,
                detector,
                keypoint_detector,
                hand_projections,
                renderer,
                device,
            )

        # img = adjust_white_balance(img)
        
        output_dir = os.path.join(output_base_dir, f"frames_{session_index}")
        os.makedirs(output_dir, exist_ok=True)
        frame_index = len(os.listdir(output_dir))
        frame_path = os.path.join(output_dir, f"frame_{frame_index:05d}.png")
        cv2.imwrite(frame_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def main():
    print("Initializing HAMER model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    model = model.to(device)
    model.eval()
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    vrs_path = "/data/borui/dataset/v2/P015_S1_002/ego/P015_S1_002.vrs"  # 替换为你的vrs文件路径
    mps_folder = "/data/borui/dataset/v2/P015_S1_002/ego/mps_P015_S1_002_vrs"  # 替换为你的mps文件夹路径
    
    output_base_dir = "./output_frames"  # 所有帧的根输出目录

    cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    #
    device1=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    cpm = ViTPoseModel(device1)

    provider = data_provider.create_vrs_data_provider(vrs_path)
    device_calibration = provider.get_device_calibration()
    T_device_CPF = device_calibration.get_transform_device_cpf()
    rgb_stream_id = StreamId("214-1")
    
    rgb_stream_label = provider.get_label_from_stream_id(rgb_stream_id)
    rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)

    rgb_linear_camera_calibration = calibration.get_linear_camera_calibration(
        int(rgb_camera_calibration.get_image_size()[0]),
        int(rgb_camera_calibration.get_image_size()[1]),
        rgb_camera_calibration.get_focal_lengths()[0],
        "pinhole",
        rgb_camera_calibration.get_transform_device_camera(),
    )
    rgb_rotated_linear_camera_calibration = calibration.rotate_camera_calib_cw90deg(rgb_linear_camera_calibration)
    camera_calibration = rgb_rotated_linear_camera_calibration

    def post_process_image(img):
        img = calibration.distort_by_calibration(
            img,
            rgb_linear_camera_calibration,
            rgb_camera_calibration,
        )
        img = np.rot90(img, k=3)
        return img

    mps_data_paths_provider = MpsDataPathsProvider(mps_folder)
    mps_data_paths = mps_data_paths_provider.get_data_paths()

    closed_loop_traj = mps.read_closed_loop_trajectory(str(mps_data_paths.slam.closed_loop_trajectory))
    eyegaze_data = mps.read_eyegaze(mps_data_paths.eyegaze.general_eyegaze)
    wrist_and_palm_poses = mps.hand_tracking.read_wrist_and_palm_poses(mps_data_paths.hand_tracking.wrist_and_palm_poses)

    deliver_option = provider.get_default_deliver_queued_options()
    deliver_option.deactivate_stream_all()
    deliver_option.activate_stream(rgb_stream_id)
    rgb_frame_count = provider.get_num_data(rgb_stream_id)

    progress_bar = tqdm(total=rgb_frame_count)

    for data in provider.deliver_queued_sensor_data(deliver_option):
        device_time_ns = data.get_time_ns(TimeDomain.DEVICE_TIME)
        # print(device_time_ns)
        # for pose in closed_loop_traj:
        #     query_timestamp_ns = int(pose.tracking_timestamp.total_seconds() * 1e9)
        pose_info = get_nearest_pose(closed_loop_traj, device_time_ns)
        if pose_info:
            
            T_world_device = pose_info.transform_world_device
            # print(T_world_device.to_matrix())
                # print(T_world_device.to_matrix())
            hand_projections = log_hand_tracking(
                wrist_and_palm_poses,
                device_time_ns,
                camera_calibration,
                rgb_stream_label,
                1,  # 替换为你的down_sampling_factor
                closed_loop_traj=closed_loop_traj
            )
            # print(f"hand_projections:{hand_projections}")
            log_RGB_image(
                data,
                1,
                75,
                rgb_stream_label,
                output_base_dir,
                camera_calibration,
                session_index=1,
                hand_projections=hand_projections,
                postprocess_image=post_process_image,
                hamer_model=model,
                model_cfg=model_cfg,     # 新增
                detector=detector,        # 新增
                keypoint_detector=cpm,  # 新增
                renderer=renderer,
                device=device,
            )
if __name__ == "__main__":
    main()