import cv2
import numpy as np
import json
from typing import Optional, List
import torch
from vitpose_model import ViTPoseModel
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
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
from projectaria_tools.core import mps
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
    T_device_camera = camera_calibration.get_transform_device_camera()
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
    if not wrist_and_palm_poses:
        return json.dumps({"left_hand": [], "right_hand": []}, indent=2)

    wrist_and_palm_pose = get_nearest_wrist_and_palm_pose(wrist_and_palm_poses, device_time_ns)
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
    if wrist_and_palm_pose.left_hand and wrist_and_palm_pose.left_hand.confidence > 0:
        left_wrist_point_device = wrist_and_palm_pose.left_hand.wrist_position_device
        left_wrist_pixel = get_camera_projection_from_device_point(left_wrist_point_device, rgb_camera_calibration)

        if left_wrist_pixel is not None:
            scaled_point = [p / down_sampling_factor for p in left_wrist_pixel]
            left_hand_points.append({"wrist_point": scaled_point})

    # Process right hand
    if wrist_and_palm_pose.right_hand and wrist_and_palm_pose.right_hand.confidence > 0:
        right_wrist_point_device = wrist_and_palm_pose.right_hand.wrist_position_device
        right_wrist_pixel = get_camera_projection_from_device_point(right_wrist_point_device, rgb_camera_calibration)

        if right_wrist_pixel is not None:
            scaled_point = [p / down_sampling_factor for p in right_wrist_pixel]
            right_hand_points.append({"wrist_point": scaled_point})

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
                left_hand_points.append({"wrist_point": future_scaled_point})

        # Process future right hand points
        if future_wrist_and_palm_pose.right_hand and future_wrist_and_palm_pose.right_hand.confidence > 0:
            future_right_wrist_point_device = future_wrist_and_palm_pose.right_hand.wrist_position_device
            future_right_wrist_point_world = T_world_device_future @ future_right_wrist_point_device
            future_right_wrist_point_device_first = T_world_device_current.inverse() @ future_right_wrist_point_world
            future_right_wrist_pixel = get_camera_projection_from_device_point(future_right_wrist_point_device_first, rgb_camera_calibration)

            if future_right_wrist_pixel is not None:
                future_scaled_point = [p / down_sampling_factor for p in future_right_wrist_pixel]
                right_hand_points.append({"wrist_point": future_scaled_point})

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


def log_RGB_image(
    data,
    down_sampling_factor: int,
    jpeg_quality: int,
    rgb_stream_label: str,
    output_base_dir: str,
    session_index: int,
    hand_projections: Optional[List[np.ndarray]] = None,
    
    postprocess_image = lambda img: img,
    # detector: Optional[DefaultPredictor_Lazy] = None,
    # cpm: Optional[ViTPoseModel] = None,
):
    output_base_dir = "./output_frames"  
    if data.sensor_data_type() == SensorDataType.IMAGE:
        img = data.image_data_and_record()[0].to_numpy_array()
        img = postprocess_image(img)
        if down_sampling_factor > 1:
            img = img[::down_sampling_factor, ::down_sampling_factor]
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)

        if hand_projections is not None:
            hand_data = json.loads(hand_projections)

            left_hand_points = hand_data.get("left_hand", [])
            left_wrist_points = [point["wrist_point"] for point in left_hand_points if "wrist_point" in point]

            right_hand_points = hand_data.get("right_hand", [])
            right_wrist_points = [point["wrist_point"] for point in right_hand_points if "wrist_point" in point]
            # print(left_wrist_points)
            # print(right_wrist_points)
            if left_wrist_points:
                for point in left_wrist_points:
                    wrist_point = point['wrist_point'] 
                    # print(right_wrist_points[i])
                    x, y = wrist_point[0], wrist_point[1]
                    cv2.circle(img, (int(x), int(y)), 1, (255, 255, 0), -1)

                for i in range(1, len(left_wrist_points)):
                    # print(right_wrist_points[i]['wrist_point'])
                    wrist_point=left_wrist_points[i]['wrist_point']
                    x, y = wrist_point[0], wrist_point[1]
                    x_prev, y_prev = left_wrist_points[i - 1]['wrist_point'][0], left_wrist_points[i - 1]['wrist_point'][1]
                    cv2.line(img, (int(x_prev), int(y_prev)), (int(x), int(y)), (255, 255, 0), 1)

            if right_wrist_points:
                for point in right_wrist_points:
                    wrist_point = point['wrist_point'] 
                    # print(right_wrist_points[i])
                    x, y = wrist_point[0], wrist_point[1]
                    cv2.circle(img, (int(x), int(y)), 1, (255, 255, 0), -1)

                for i in range(1, len(right_wrist_points)):
                    # print(right_wrist_points[i]['wrist_point'])
                    wrist_point=right_wrist_points[i]['wrist_point']
                    x, y = wrist_point[0], wrist_point[1]
                    x_prev, y_prev = right_wrist_points[i - 1]['wrist_point'][0], right_wrist_points[i - 1]['wrist_point'][1]
                    cv2.line(img, (int(x_prev), int(y_prev)), (int(x), int(y)), (255, 255, 0), 1)

        img = adjust_white_balance(img)

        output_dir = os.path.join(output_base_dir, f"frames_{session_index}")
        os.makedirs(output_dir, exist_ok=True)
        frame_index = len(os.listdir(output_dir))
        frame_path = os.path.join(output_dir, f"frame_{frame_index:05d}.png")
        cv2.imwrite(frame_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # rr.log(
        #     f"world/device/{rgb_stream_label}",
        #     rr.Image(img).compress(jpeg_quality=jpeg_quality),
        # )
        
def main():
    #####################################
    # 初始化detector和ViTPose模型
    #####################################
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cpm = ViTPoseModel(device)  # 初始化ViTPose模型

    # cfg_path = Path("./hamer/hamer/configs/cascade_mask_rcnn_vitdet_h_75ep.py")
    # detectron2_cfg = LazyConfig.load(str(cfg_path))
    # detectron2_cfg.train.device = "cuda"  # 不要写成 "cuda:0"

    # detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    # detector = DefaultPredictor_Lazy(detectron2_cfg)

    args = parse_args()
    base_folder = "dataset"  # 替换为你的顶层数据集文件夹路径

    # 自动匹配 .vrs 和 hand_tracking 文件
    if not args.vrs or not args.hands:
        matched_files = find_vrs_and_handtracking(base_folder)
        if not matched_files:
            print("No matched VRS and handtracking files found!")
            return
    else:
        matched_files = [(Path(args.vrs), Path(args.hands))]  # 使用命令行指定的路径

    vrs_path = "collected_data/Cook-egg.vrs"  # 替换为你的vrs文件路径
    mps_folder = "/data/borui/collected_data/mps_Cook-egg_vrs"  # 替换为你的mps文件夹路径
    
    output_base_dir = "./output_frames"  # 所有帧的根输出目录

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
            print(T_world_device.to_matrix())
                # print(T_world_device.to_matrix())
            hand_projections = log_hand_tracking(
                wrist_and_palm_poses,
                device_time_ns,
                camera_calibration,
                rgb_stream_label,
                1,  # 替换为你的down_sampling_factor
                closed_loop_traj=closed_loop_traj
            )

            log_RGB_image(
                data,
                1,  # 替换为你的down_sampling_factor  
                75,  # 替换为你的jpeg_quality
                rgb_stream_label,
                session_index=1,  # 替换为你的session_index
                hand_projections=hand_projections,
                output_base_dir=output_base_dir,
                postprocess_image=post_process_image,
            )

if __name__ == "__main__":
    main()