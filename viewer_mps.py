# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# ... (省略license内容)
import argparse
import os
import cv2
from pathlib import Path
from typing import Callable, List, Optional
import json
import numpy as np
import rerun as rr

from projectaria_tools.core import calibration, data_provider, mps, image
from projectaria_tools.core.calibration import CameraCalibration, DeviceCalibration
from projectaria_tools.core.mps import MpsDataPathsProvider
from projectaria_tools.core.mps.utils import (
    filter_points_from_confidence,
    filter_points_from_count,
    get_gaze_vector_reprojection,
    get_nearest_eye_gaze,
    get_nearest_pose,
    get_nearest_wrist_and_palm_pose,
)
from projectaria_tools.core.sensor_data import SensorData, SensorDataType, TimeDomain
from projectaria_tools.core.sophus import SE3
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.utils.rerun_helpers import AriaGlassesOutline, ToTransform3D
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
# 以下为手部关键点检测和绘制需要的依赖
import torch
from vitpose_model import ViTPoseModel
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
from hamer.datasets.vitdet_dataset import DEFAULT_MEAN, DEFAULT_STD

WRIST_PALM_TIME_DIFFERENCE_THRESHOLD_NS: int = 2e8
WRIST_PALM_COLOR: List[int] = [255, 64, 0]
NORMAL_VIS_LEN = 0.03  # in meters
NORMAL_VIS_LEN_2D = 30.0  # in pixels
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
    point: np.ndarray, camera_calibration: CameraCalibration
) -> Optional[np.ndarray]:
    T_device_camera = camera_calibration.get_transform_device_camera()
    # print(dir(T_device_camera))
    return camera_calibration.project(T_device_camera.inverse() @ point)


def log_device_trajectory(trajectory_files: List[str]) -> None:
    print("Loading and logging trajectory(ies)...")
    trajectory_list_size = len(trajectory_files)
    i = 0
    for trajectory_file in trajectory_files:
        print(f"Loading: {trajectory_file}")
        trajectory_data = mps.read_closed_loop_trajectory(trajectory_file)
        device_trajectory = [
            it.transform_world_device.translation()[0] for it in trajectory_data
        ][0::80]

        entity_path = (
            "world/device_trajectory"
            if trajectory_list_size == 1
            else f"world/device_trajectory_{i}"
        )
        rr.log(entity_path, rr.LineStrips3D(device_trajectory, radii=0.008), static=True)
        print(f"Showing: {trajectory_file} as {entity_path}")
        i += 1


def log_point_clouds(points_files: List[str]) -> None:
    print("Loading and logging point cloud(s)...")
    point_cloud_list_size = len(points_files)
    i = 0
    for points_file in points_files:
        points_data = mps.read_global_point_cloud(points_file)
        points_data = filter_points_from_confidence(points_data)
        points_data_down_sampled = filter_points_from_count(
            points_data, 500_000 if point_cloud_list_size == 1 else 20_000
        )
        point_positions = [it.position_world for it in points_data_down_sampled]

        entity_path = (
            "world/points" if point_cloud_list_size == 1 else f"world/points_{i}"
        )
        rr.log(entity_path, rr.Points3D(point_positions, radii=0.006), static=True)
        print(f"Showing: {points_file} as {entity_path}")
        i += 1


def log_RGB_camera_calibration(
    rgb_camera_calibration: CameraCalibration,
    rgb_stream_label: str,
    down_sampling_factor: int,
) -> None:
    rr.log(
        f"world/device/{rgb_stream_label}",
        rr.Pinhole(
            resolution=[
                rgb_camera_calibration.get_image_size()[0] / down_sampling_factor,
                rgb_camera_calibration.get_image_size()[1] / down_sampling_factor,
            ],
            focal_length=float(
                rgb_camera_calibration.get_focal_lengths()[0] / down_sampling_factor
            ),
        ),
        static=True,
    )


def log_Aria_glasses_outline(device_calibration: DeviceCalibration) -> None:
    aria_glasses_point_outline = AriaGlassesOutline(device_calibration)
    rr.log(
        "world/device/glasses_outline",
        rr.LineStrips3D([aria_glasses_point_outline]),
        static=True,
    )


def log_camera_pose(
    trajectory_data: List[mps.ClosedLoopTrajectoryPose],
    device_time_ns: int,
    rgb_camera_calibration: CameraCalibration,
    rgb_stream_label: str,
) -> Optional[SE3]:
    print(trajectory_data)
    if trajectory_data:
        pose_info = get_nearest_pose(trajectory_data, device_time_ns)
        T_world_device = pose_info.transform_world_device
        T_device_camera = rgb_camera_calibration.get_transform_device_camera()
        print(T_world_device.to_matrix())
        rr.log("world/device", ToTransform3D(T_world_device, False))
        rr.log(
            f"world/device/{rgb_stream_label}",
            ToTransform3D(T_device_camera, False),
        )
        return T_world_device
    return None


def adjust_white_balance(image):
    result = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    avg_a = np.mean(result[:, :, 1])
    avg_b = np.mean(result[:, :, 2])
    result[:, :, 1] = np.clip(result[:, :, 1] - (avg_a - 128), 0, 255).astype(np.uint8)
    result[:, :, 2] = np.clip(result[:, :, 2] - (avg_b - 128), 0, 255).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
    return result


def log_eye_gaze(
    eyegaze_data: List[mps.EyeGaze],
    device_time_ns: int,
    T_device_CPF: SE3,
    rgb_stream_label: str,
    device_calibration: DeviceCalibration,
    rgb_camera_calibration: CameraCalibration,
    down_sampling_factor: int,
    make_upright: bool = False,
) -> None:
    logged_eyegaze: bool = False
    if eyegaze_data:
        eye_gaze = get_nearest_eye_gaze(eyegaze_data, device_time_ns)
        if eye_gaze:
            depth_m = eye_gaze.depth or 1.0
            gaze_vector_in_cpf = mps.get_eyegaze_point_at_depth(
                eye_gaze.yaw, eye_gaze.pitch, depth_m
            )
            rr.log(
                "world/device/eye-gaze",
                rr.Arrows3D(
                    origins=[T_device_CPF @ [0, 0, 0]],
                    vectors=[T_device_CPF @ gaze_vector_in_cpf],
                    colors=[[255, 0, 255]],
                ),
            )
            gaze_projection = get_gaze_vector_reprojection(
                eye_gaze=eye_gaze,
                stream_id_label=rgb_stream_label,
                device_calibration=device_calibration,
                camera_calibration=rgb_camera_calibration,
                depth_m=depth_m,
                make_upright=make_upright,
            )
            if gaze_projection is not None:
                rr.log(
                    f"world/device/{rgb_stream_label}/eye-gaze_projection",
                    rr.Points2D(
                        gaze_projection / down_sampling_factor,
                        radii=4,
                    ),
                )
                logged_eyegaze = True
    if not logged_eyegaze:
        rr.log("world/device/eye-gaze", rr.Clear.flat())
        rr.log(
            f"world/device/{rgb_stream_label}/eye-gaze_projection",
            rr.Clear.flat(),
        )


def log_hand_tracking(
    wrist_and_palm_poses,
    device_time_ns: int,
    rgb_camera_calibration,
    rgb_stream_label: str,
    down_sampling_factor: int,
    future_interval_seconds: float = 1.0,
    num_samples: int = 10,
) -> str:
    if not wrist_and_palm_poses:
        return json.dumps({"left_hand": [], "right_hand": []}, indent=2)

    future_interval_ns = int(future_interval_seconds * 1e9)
    t_start = device_time_ns
    t_end = device_time_ns + future_interval_ns
    future_timestamps = np.linspace(t_start, t_end, num_samples)

    left_hand_points = []
    right_hand_points = []

    for t_future in future_timestamps:
        wrist_and_palm_pose = get_nearest_wrist_and_palm_pose(wrist_and_palm_poses, t_future)
        if wrist_and_palm_pose is None:
            continue

        if (
            wrist_and_palm_pose
            and abs(wrist_and_palm_pose.tracking_timestamp.total_seconds() * 1e9 - t_future)
            < WRIST_PALM_TIME_DIFFERENCE_THRESHOLD_NS
        ):
            for one_side_pose in [wrist_and_palm_pose.right_hand, wrist_and_palm_pose.left_hand]:
                if one_side_pose and one_side_pose.confidence > 0:
                    wrist_points = [one_side_pose.wrist_position_device]
                    wrist_pixels = [
                        get_camera_projection_from_device_point(point, rgb_camera_calibration)
                        for point in wrist_points
                    ]
                    for wrist_pixel in wrist_pixels:
                        if wrist_pixel is not None:
                            valid_points = [p for p in wrist_pixel if p is not None]
                            scaled_points = [vp / down_sampling_factor for vp in valid_points]
                            if one_side_pose == wrist_and_palm_pose.right_hand:
                                right_hand_points.extend(scaled_points)
                            else:
                                left_hand_points.extend(scaled_points)

    result = {
    "left_hand": [
        {"wrist_point": point} for point in left_hand_points
    ] if len(left_hand_points) >= 2 else [],
    "right_hand": [
        {"wrist_point": point} for point in right_hand_points
    ] if len(right_hand_points) >= 2 else [],
}


    result_json = json.dumps(result, indent=2)
    return result_json


#############################
# 以下为新增加的检测与绘制函数
#############################

def draw_keypoints_and_connections(image, keypoints, confidence_threshold=0.8):
    """
    在图像上绘制关键点并连接关键点
    """
    # 手部关键点连接顺序（21个关键点，典型手部COCO格式）
    connections = [
        [0, 1, 2, 3, 4],  # 拇指
        [0, 5, 6, 7, 8],  # 食指
        [0, 9, 10, 11, 12],  # 中指
        [0, 13, 14, 15, 16],  # 无名指
        [0, 17, 18, 19, 20]  # 小指
    ]

    # image_2=iamge.copy()
    for conn in connections:
        for i in range(len(conn) - 1):
            kp1, kp2 = keypoints[conn[i]], keypoints[conn[i + 1]]
            if kp1[2] > confidence_threshold and kp2[2] > confidence_threshold:
                x1, y1 = int(kp1[0]), int(kp1[1])
                x2, y2 = int(kp2[0]), int(kp2[1])
                cv2.line(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)
        # for kp in keypoints:
        # if kp[2] > confidence_threshold:
        #     x, y = int(kp[0]), int(kp[1])
        #     cv2.circle(image, (x, y), radius=1, color=(0, 255, 0), thickness=-1)
    #no hamer
    # for conn in connections:
    #     for i in range(len(conn) - 1):
    #         kp1, kp2 = keypoints[conn[i]], keypoints[conn[i + 1]]
    #         if kp1[2] > confidence_threshold and kp2[2] > confidence_threshold:
    #             x1, y1 = int(kp1[0]), int(kp1[1])
    #             x2, y2 = int(kp2[0]), int(kp2[1])
    #             cv2.line(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)


    return image

def detect_and_draw_hands(img: np.ndarray, detector: DefaultPredictor_Lazy, cpm: ViTPoseModel, hand_projections: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """
    使用人体检测器和ViTPose来检测图像中的人体和手部关键点，并在图像上绘制。
    """
    img_cv2 = img.copy()
    img_bgr = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)  # detector使用BGR
    det_out = detector(img_bgr)
    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.8)
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

    if pred_bboxes.shape[0] == 0:
        # 没有检测到人体则直接返回原图
        return img

    # vitpose 推断关键点
    vitposes_out = cpm.predict_pose(
        img_cv2,
        [np.concatenate([pred_bboxes, det_instances.scores[valid_idx, None].cpu().numpy()], axis=1)],
    )
    is_right_arr = []

        # print(left_wrist_points)
            # 为左右手用不同颜色打点（例如左手用绿色，右手用蓝色）
        # if left_wrist_points is not None:
        #     for (x, y) in left_wrist_points:
        #         cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)
        # if right_wrist_points is not None:
        #     for (x, y) in right_wrist_points:
        #         cv2.circle(img, (int(x), int(y)), 1, (255, 0, 0), -1)
    # 对检测到的人进行手部关键点绘制
    for vitposes in vitposes_out:
        left_hand_keyp = vitposes['keypoints'][-42:-21]
        right_hand_keyp = vitposes['keypoints'][-21:]
        if hand_projections is not None:
            # hand_projections是JSON字符串，先解析成字典
            hand_data = json.loads(hand_projections)
            # print(hand_projections)
                # 左手点数据
            left_wrist_points = hand_data.get("left_hand", {}).get("wrist_points", [])
                # 右手点数据
            right_wrist_points = hand_data.get("right_hand", {}).get("wrist_points", [])
            #这里是替换手腕点
            # if len(left_wrist_points) > 0 and len(left_wrist_points[0]) == 2:
            #     # test hamer
            #     lx, ly = left_wrist_points[0]
            #     # 将left_hand_keyp[0]的坐标替换
            #     left_hand_keyp[0][0] = lx
            #     left_hand_keyp[0][1] = ly
            #     left_hand_keyp[0][2] = 1.0  # 设置高置信度

            # if len(right_wrist_points) > 0 and len(right_wrist_points[0]) == 2:
            #     rx, ry = right_wrist_points[0]
            #     # 将right_hand_keyp[0]的坐标替换
            #     right_hand_keyp[0][0] = rx
            #     right_hand_keyp[0][1] = ry
            #     right_hand_keyp[0][2] = 1.0  # 设置高置信度
                    

        # 现在使用更新后的关键点继续绘制
        img_cv2 = draw_keypoints_and_connections(img_cv2, left_hand_keyp)
        img_cv2 = draw_keypoints_and_connections(img_cv2, right_hand_keyp)


    # 返回处理过的RGB图像（ViTPose内部处理为RGB，这里最终需要返回RGB格式以保持一致）
    return img_cv2 

####################################
# 修改log_RGB_image函数，调用detect_and_draw_hands
####################################

def log_RGB_image(
    data: SensorData,
    down_sampling_factor: int,
    jpeg_quality: int,
    rgb_stream_label: str,
    output_base_dir: str,
    session_index: int,
    hand_projections: Optional[List[np.ndarray]] = None,
    
    postprocess_image: Callable[[np.ndarray], np.ndarray] = lambda img: img,
    detector: Optional[DefaultPredictor_Lazy] = None,
    cpm: Optional[ViTPoseModel] = None,
) -> None:
    output_base_dir = "./output_frames"  # 所有帧的根输出目录
    if data.sensor_data_type() == SensorDataType.IMAGE:
        img = data.image_data_and_record()[0].to_numpy_array()
        img = postprocess_image(img)
        if down_sampling_factor > 1:
            img = img[::down_sampling_factor, ::down_sampling_factor]
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)
        # print(hand_projections)
        # 手腕点轨迹
        if hand_projections is not None:
            # 解析 JSON 字符串为字典
            hand_data = json.loads(hand_projections)

            # 获取左手点数据
            left_hand_points = hand_data.get("left_hand", [])
            left_wrist_points = [point["wrist_point"] for point in left_hand_points if "wrist_point" in point]

            # 获取右手点数据
            right_hand_points = hand_data.get("right_hand", [])
            right_wrist_points = [point["wrist_point"] for point in right_hand_points if "wrist_point" in point]

            # 打印数据以确认
            # print("Left wrist points:", left_wrist_points)
            # print("Right wrist points:", right_wrist_points)

            # 绘制未来一秒的点
            # 绘制未来一秒的点
            if left_wrist_points:
                # 重组为二维点的列表
                reshaped_left_wrist_points = [
                    left_wrist_points[i:i + 2] for i in range(0, len(left_wrist_points), 2)
                ]

                # 绘制左手的点并连线
                for i in range(len(reshaped_left_wrist_points)):
                    x, y = reshaped_left_wrist_points[i]
                    cv2.circle(img, (int(x), int(y)), 1, (255, 0, 0), -1)  # 绘制点

                    # 连线到下一个点
                    if i > 0:
                        x_prev, y_prev = reshaped_left_wrist_points[i - 1]
                        cv2.line(img, (int(x_prev), int(y_prev)), (int(x), int(y)), (255, 0, 0), 1)

            if right_wrist_points:
                # 重组为二维点的列表
                reshaped_right_wrist_points = [
                    right_wrist_points[i:i + 2] for i in range(0, len(right_wrist_points), 2)
                ]

                # 绘制右手的点并连线
                for i in range(len(reshaped_right_wrist_points)):
                    x, y = reshaped_right_wrist_points[i]
                    cv2.circle(img, (int(x), int(y)), 1, (255, 255, 0), -1)  # 绘制点

                    # 连线到下一个点
                    if i > 0:
                        x_prev, y_prev = reshaped_right_wrist_points[i - 1]
                        cv2.line(img, (int(x_prev), int(y_prev)), (int(x), int(y)), (255, 255, 0), 1)


        # 调整白平衡
        img = adjust_white_balance(img)

        # 在这里调用手部关键点检测函数
        # 如果detector和cpm已初始化，执行手部关键点检测
        # if detector is not None and cpm is not None:
        #     img = detect_and_draw_hands(img, detector, cpm, hand_projections)
        output_dir = os.path.join(output_base_dir, f"frames_{session_index}")
        os.makedirs(output_dir, exist_ok=True)
        frame_index = len(os.listdir(output_dir))
        frame_path = os.path.join(output_dir, f"frame_{frame_index:05d}.png")
        cv2.imwrite(frame_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        rr.log(
            f"world/device/{rgb_stream_label}",
            rr.Image(img).compress(jpeg_quality=jpeg_quality),
        )


def create_video_from_frames(frame_dir: str, output_video_path: str, fps: int = 30) -> None:
    frames = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".png")])
    if not frames:
        raise ValueError("No frames found in directory!")
    first_frame = cv2.imread(frames[0])
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_path in frames:
        frame = cv2.imread(frame_path)
        video_writer.write(frame)
    video_writer.release()


def main():
    #####################################
    # 初始化detector和ViTPose模型
    #####################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpm = ViTPoseModel(device)  # 初始化ViTPose模型

    cfg_path = Path("./hamer/hamer/configs/cascade_mask_rcnn_vitdet_h_75ep.py")
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.device = "cuda"  # 不要写成 "cuda:0"

    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    # device_ids = [0, 1]
    # cpm = torch.nn.DataParallel(cpm, device_ids=device_ids).to(f"cuda:{device_ids[0]}")

    # detector.model = torch.nn.DataParallel(detector.model, device_ids=[0,1]).to(device)

    args = parse_args()
    base_folder = "test"  # 替换为你的顶层数据集文件夹路径

    # 自动匹配 .vrs 和 hand_tracking 文件
    if not args.vrs or not args.hands:
        matched_files = find_vrs_and_handtracking(base_folder)
        if not matched_files:
            print("No matched VRS and handtracking files found!")
            return
    else:
        matched_files = [(Path(args.vrs), Path(args.hands))]  # 使用命令行指定的路径

    output_base_dir = "./output_frames"  # 所有帧的根输出目录

    # # 遍历匹配的 (vrs_file, handtracking_file)
    
    if args.vrs:
        vrs_folder_path = os.path.dirname(args.vrs)
        if args.points is None and args.eyegaze is None and args.trajectory is None:
            if args.mps_folder:
                mps_data_paths_provider = MpsDataPathsProvider(args.mps_folder)
            else:
                mps_data_paths_provider = MpsDataPathsProvider(str(Path(vrs_folder_path + "/mps")))
            mps_data_paths = mps_data_paths_provider.get_data_paths()

            if not args.trajectory and os.path.exists(mps_data_paths.slam.closed_loop_trajectory):
                args.trajectory = [str(mps_data_paths.slam.closed_loop_trajectory)]

            if not args.points and os.path.exists(mps_data_paths.slam.semidense_points):
                args.points = [str(mps_data_paths.slam.semidense_points)]

            if not args.eyegaze and os.path.exists(mps_data_paths.eyegaze.personalized_eyegaze):
                args.eyegaze = mps_data_paths.eyegaze.personalized_eyegaze
            if not args.eyegaze and os.path.exists(mps_data_paths.eyegaze.general_eyegaze):
                args.eyegaze = mps_data_paths.eyegaze.general_eyegaze

            if not args.hands and os.path.exists(mps_data_paths.hand_tracking.wrist_and_palm_poses):
                args.hands = mps_data_paths.hand_tracking.wrist_and_palm_poses

    mps_data_available = args.trajectory or args.points or args.eyegaze or args.hands
    print(
        f"""
    Trying to load the following list of files:
    - vrs: {args.vrs}
    - trajectory/closed_loop_trajectory: {args.trajectory}
    - trajectory/point_cloud: {args.points}
    - eye_gaze/general_eye_gaze: {args.eyegaze}
    - hand_tracking/wrist_and_palm_poses: {args.hands}
    """
    )

    if mps_data_available is None and args.vrs is None:
        print("Nothing to display.")
        exit(1)
    for session_index, (vrs_file, handtracking_file) in enumerate(matched_files, start=1):
        print(f"\nProcessing VRS: {vrs_file} with HandTracking: {handtracking_file}")

        # 设置当前的 VRS 和 HandTracking 文件路径
        args.vrs = str(vrs_file)
        args.hands = str(handtracking_file)
        rr.init("MPS Data Viewer", spawn=(not args.rrd_output_path))
        if args.rrd_output_path:
            print(f"Saving .rrd file to {args.rrd_output_path}")
            rr.save(args.rrd_output_path)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        # if args.trajectory:
        #     log_device_trajectory(args.trajectory)

        # if args.points:
        #     log_point_clouds(args.points)

        # if not args.vrs:
        #     return

        provider = data_provider.create_vrs_data_provider(args.vrs)
        device_calibration = provider.get_device_calibration()
        T_device_CPF = device_calibration.get_transform_device_cpf()
        rgb_stream_id = StreamId("214-1")
        rgb_stream_label = provider.get_label_from_stream_id(rgb_stream_id)
        rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)

        if not args.no_rectify_image:
            rgb_linear_camera_calibration = calibration.get_linear_camera_calibration(
                int(rgb_camera_calibration.get_image_size()[0]),
                int(rgb_camera_calibration.get_image_size()[1]),
                rgb_camera_calibration.get_focal_lengths()[0],
                "pinhole",
                rgb_camera_calibration.get_transform_device_camera(),
            )
            if not args.no_rotate_image_upright:
                rgb_rotated_linear_camera_calibration = calibration.rotate_camera_calib_cw90deg(rgb_linear_camera_calibration)
                camera_calibration = rgb_rotated_linear_camera_calibration
            else:
                camera_calibration = rgb_linear_camera_calibration
        else:
            if args.no_rotate_image_upright:
                camera_calibration = rgb_camera_calibration
            else:
                raise NotImplementedError(
                    "Upright rotation without rectification not implemented."
                )

        def post_process_image(img):
            if not args.no_rectify_image:
                img = calibration.distort_by_calibration(
                    img,
                    rgb_linear_camera_calibration,
                    rgb_camera_calibration,
                )
                if not args.no_rotate_image_upright:
                    img = np.rot90(img, k=3)
            return img

        trajectory_data = mps.read_closed_loop_trajectory(str(args.trajectory[0])) if args.trajectory else None
        eyegaze_data = mps.read_eyegaze(args.eyegaze) if args.eyegaze else None
        wrist_and_palm_poses = mps.hand_tracking.read_wrist_and_palm_poses(args.hands) if args.hands else None

        log_RGB_camera_calibration(rgb_camera_calibration, rgb_stream_label, args.down_sampling_factor)
        log_Aria_glasses_outline(device_calibration)

        deliver_option = provider.get_default_deliver_queued_options()
        deliver_option.deactivate_stream_all()
        deliver_option.activate_stream(rgb_stream_id)
        rgb_frame_count = provider.get_num_data(rgb_stream_id)

        progress_bar = tqdm(total=rgb_frame_count)

        


        for data in provider.deliver_queued_sensor_data(deliver_option):
            device_time_ns = data.get_time_ns(TimeDomain.DEVICE_TIME)
            rr.set_time_nanos("device_time", device_time_ns)
            rr.set_time_sequence("timestamp", device_time_ns)
            progress_bar.update(1)

            T_world_device = log_camera_pose(
                trajectory_data,
                device_time_ns,
                camera_calibration,
                rgb_stream_label,
            )

            hand_projections = log_hand_tracking(
                wrist_and_palm_poses,
                device_time_ns,
                camera_calibration,
                rgb_stream_label,
                args.down_sampling_factor,
            )

            log_RGB_image(
                data,
                args.down_sampling_factor,
                args.jpeg_quality,
                rgb_stream_label,
                session_index=session_index,
                hand_projections=hand_projections,
                output_base_dir=output_base_dir,
                postprocess_image=post_process_image,
                detector=detector,
                cpm=cpm,
            )

            log_eye_gaze(
                eyegaze_data,
                device_time_ns,
                T_device_CPF,
                rgb_stream_label,
                device_calibration,
                camera_calibration,
                args.down_sampling_factor,
                not args.no_rotate_image_upright,
            )

if __name__ == "__main__":
    main()