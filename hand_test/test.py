import cv2
import numpy as np
import json
from typing import Optional, List, Tuple, Dict
import torch
from pathlib import Path
import os
from queue import Queue, Empty
import threading
from dataclasses import dataclass
from threading import Event, Lock
import time
from tqdm import tqdm
from collections import deque
from projectaria_tools.core import calibration, data_provider, mps
from projectaria_tools.core.calibration import CameraCalibration
from projectaria_tools.core.mps import MpsDataPathsProvider
from projectaria_tools.core.mps.utils import get_nearest_pose, get_nearest_wrist_and_palm_pose
from projectaria_tools.core.sensor_data import SensorData, SensorDataType, TimeDomain
from projectaria_tools.core.stream_id import StreamId

import hamer
from hamer.models import HAMER, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from hamer.utils import recursive_to
from detectron2.config import LazyConfig
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer.utils.render_openpose import render_openpose
from hamer.utils.geometry import perspective_projection
from vitpose_model import ViTPoseModel

@dataclass
class HandTrajectoryData:
    """存储手部轨迹数据"""
    current_wrist_point: Optional[List[float]]
    trajectory_points: List[List[float]]
    confidence: float

class HandTrajectoryBuffer:
    """手部轨迹数据缓冲区，包含同步机制"""
    def __init__(self):
        self.lock = Lock()
        self.left_hand_data: Optional[HandTrajectoryData] = None
        self.right_hand_data: Optional[HandTrajectoryData] = None
        
    def update(self, hand_projections: str):
        """更新轨迹数据"""
        with self.lock:
            try:
                data = json.loads(hand_projections)
                
                # 处理左手数据
                left_hand_points = data.get("left_hand", [])
                if left_hand_points:
                    self.left_hand_data = HandTrajectoryData(
                        current_wrist_point=left_hand_points[0]["wrist_point"],
                        trajectory_points=[point["wrist_point"] for point in left_hand_points],
                        confidence=1.0  # 可以根据需要调整置信度计算
                    )
                else:
                    self.left_hand_data = None
                    
                # 处理右手数据
                right_hand_points = data.get("right_hand", [])
                if right_hand_points:
                    self.right_hand_data = HandTrajectoryData(
                        current_wrist_point=right_hand_points[0]["wrist_point"],
                        trajectory_points=[point["wrist_point"] for point in right_hand_points],
                        confidence=1.0
                    )
                else:
                    self.right_hand_data = None
                    
            except Exception as e:
                print(f"Error updating trajectory buffer: {e}")
                
    def get_data(self) -> Tuple[Optional[HandTrajectoryData], Optional[HandTrajectoryData]]:
        """获取轨迹数据"""
        with self.lock:
            return self.left_hand_data, self.right_hand_data

@dataclass
class ProcessingTask:
    """表示一个处理任务的数据类"""
    frame_id: int
    image: np.ndarray
    device_time_ns: int
    hand_projections: str

@dataclass
class ProcessingResult:
    """表示处理结果的数据类"""
    frame_id: int
    processed_image: np.ndarray

class HandDetectionThread(threading.Thread):
    """手部检测线程"""
    def __init__(self, input_queue: Queue, output_queue: Queue, model, model_cfg, 
                 detector, keypoint_detector, device, trajectory_buffer: HandTrajectoryBuffer,
                 stop_event: Event):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.model = model
        self.model_cfg = model_cfg
        self.detector = detector
        self.keypoint_detector = keypoint_detector
        self.device = device
        self.trajectory_buffer = trajectory_buffer
        self.stop_event = stop_event
        self.openpose_indices = list(range(21))
        self.gt_indices = list(range(21))

    def process_hand_detection(self, img: np.ndarray) -> np.ndarray:
        """处理手部检测的核心逻辑"""
        img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        det_out = self.detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        # 使用关键点检测器预测姿势
        vitposes_out = self.keypoint_detector.predict_pose(
            img_cv2,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        # 获取当前的轨迹数据
        left_hand_data, right_hand_data = self.trajectory_buffer.get_data()

        bboxes = []
        is_right = []

        # 处理预测结果
        for vitposes in vitposes_out:
            left_hand_bbox = None
            right_hand_bbox = None
            left_max_conf = 0
            right_max_conf = 0
            
            # 处理左手关键点
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
            
            # 处理右手关键点
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

            # 添加有效的边界框
            if left_hand_bbox is not None:
                bboxes.append(left_hand_bbox)
                is_right.append(0)
            if right_hand_bbox is not None:
                bboxes.append(right_hand_bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            return img_cv2

        # 准备数据集
        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        
        dataset = ViTDetDataset(
            self.model_cfg, 
            img_cv2, 
            boxes, 
            right, 
            rescale_factor=2.0
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=8, 
            shuffle=False, 
            num_workers=0,
            pin_memory=True
        )

        # 处理预测结果
        all_pred_2d = []

        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model(batch)

            # 处理预测结果...
            # [原代码中的处理逻辑]
            # 替换手腕点坐标
            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(
                pred_cam, 
                box_center, 
                box_size, 
                img_size, 
                scaled_focal_length
            )

            batch_size = batch['img'].shape[0]
               
            pred_keypoints_3d = out['pred_keypoints_3d'].reshape(batch_size, -1, 3)
            for i in range(batch_size):
                current_multiplier = multiplier[i]
                pred_keypoints_3d[i,:,0] = current_multiplier * pred_keypoints_3d[i,:,0]

            out['pred_keypoints_2d'] = perspective_projection(
                pred_keypoints_3d,
                translation=pred_cam_t_full.reshape(batch_size, 3),
                focal_length=torch.tensor([[scaled_focal_length, scaled_focal_length]]),
                camera_center=torch.tensor([703.5,703.5])
            )
            
            pred_cam_t_full = pred_cam_t_full.detach().cpu().numpy()
            
            for i in range(batch_size):  # 修改这里，使用有效的循环变量
                person_id = int(batch['personid'][i])
                pred_joints = out['pred_keypoints_2d'][i].detach().cpu().numpy()
                is_right = int(batch['right'][i].cpu().numpy())
                
                # 根据是否为右手替换手腕点坐标
                if is_right and right_hand_data and right_hand_data.current_wrist_point is not None:
                    pred_joints[0, 0] = right_hand_data.current_wrist_point[0]
                    pred_joints[0, 1] = right_hand_data.current_wrist_point[1]
                elif not is_right and left_hand_data and left_hand_data.current_wrist_point is not None:
                    pred_joints[0, 0] = left_hand_data.current_wrist_point[0]
                    pred_joints[0, 1] = left_hand_data.current_wrist_point[1]

                v = np.ones((21, 1))
                pred_joints = np.concatenate((pred_joints, v), axis=-1)
                all_pred_2d.append(pred_joints)

        # 绘制关键点和轨迹
        processed_img = self.draw_keypoints_and_trajectory(
            img_cv2, 
            all_pred_2d, 
            left_hand_data, 
            right_hand_data
        )
        
        return processed_img
    def run(self):
        """线程运行方法"""
        while not self.stop_event.is_set():
            try:
                task = self.input_queue.get(timeout=1.0)
                try:
                    processed_img = self.process_hand_detection(task.image)
                    result = ProcessingResult(task.frame_id, processed_img)
                    self.output_queue.put(result)
                except Exception as e:
                    print(f"Error in processing: {e}")
                finally:
                    self.input_queue.task_done()
            except Empty:  # 从 queue 模块直接导入 Empty
                continue
            except Exception as e:
                print(f"Error in detection thread: {e}")


    def draw_keypoints_and_trajectory(
        self, 
        img: np.ndarray, 
        all_pred_2d: List[np.ndarray],
        left_hand_data: Optional[HandTrajectoryData],
        right_hand_data: Optional[HandTrajectoryData]
    ) -> np.ndarray:
        """绘制关键点和轨迹"""
        # 首先绘制轨迹
        img_copy = img.copy()
        
        # 绘制左手轨迹
        if left_hand_data and left_hand_data.trajectory_points:
            points = np.array(left_hand_data.trajectory_points, dtype=np.int32)
            for point in points:
                cv2.circle(img_copy, (int(point[0]), int(point[1])), 2, (255, 255, 0), -1)
            cv2.polylines(img_copy, [points], False, (255, 255, 0), 2, cv2.LINE_AA)
            
        # 绘制右手轨迹
        if right_hand_data and right_hand_data.trajectory_points:
            points = np.array(right_hand_data.trajectory_points, dtype=np.int32)
            for point in points:
                cv2.circle(img_copy, (int(point[0]), int(point[1])), 2, (255, 255, 0), -1)
            cv2.polylines(img_copy, [points], False, (255, 255, 0), 2, cv2.LINE_AA)

        # 绘制关键点
        input_img = img_copy.astype(np.float32)[:,:,::-1]/255.0
        input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2)
        pred_img = input_img.copy()[:,:,:-1][:,:,::-1] * 255
        
        for pred_joints in all_pred_2d:
            body_keypoints_2d = pred_joints[:21].copy()
            pred_img = render_openpose(pred_img, body_keypoints_2d)
        
        return cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)

    def run(self):
        while not self.stop_event.is_set():
            try:
                task = self.input_queue.get(timeout=1.0)
                processed_img = self.process_hand_detection(task.image)
                
                result = ProcessingResult(task.frame_id, processed_img)
                self.output_queue.put(result)
                
                self.input_queue.task_done()
            except Queue.Empty:
                continue
            except Exception as e:
                print(f"Error in detection thread: {e}")
                self.input_queue.task_done()

class HandTrackingSystem:
    def __init__(self, vrs_path: str, mps_folder: str, output_base_dir: str, num_threads: int = 2):
        """初始化手部跟踪系统"""
        self.vrs_path = vrs_path
        self.mps_folder = mps_folder
        self.output_base_dir = output_base_dir
        self.num_threads = num_threads
        
        # 初始化队列和缓冲区
        self.input_queue = Queue(maxsize=30)
        self.output_queue = Queue()
        self.trajectory_buffer = HandTrajectoryBuffer()
        
        # 初始化停止事件
        self.stop_event = Event()
        
        # 初始化其他组件
        self._init_models()
        self._init_providers()
        
        # 初始化处理线程
        self.processing_threads = []
        self._init_processing_threads()

    def _init_models(self):
        """初始化所有需要的模型和配置"""
        # 初始化设备
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_keypoint = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        
        # 初始化HAMER模型
        self.model, self.model_cfg = load_hamer(DEFAULT_CHECKPOINT)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)
        
        # 初始化检测器
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        
        # 初始化关键点检测器
        self.keypoint_detector = ViTPoseModel(self.device_keypoint)

    def _init_providers(self):
        """初始化数据提供者和相关配置"""
        # 初始化VRS数据提供者
        self.provider = data_provider.create_vrs_data_provider(self.vrs_path)
        self.device_calibration = self.provider.get_device_calibration()
        
        # 设置RGB流
        self.rgb_stream_id = StreamId("214-1")
        self.rgb_stream_label = self.provider.get_label_from_stream_id(self.rgb_stream_id)
        
        # 获取相机标定
        self._setup_camera_calibration()
        
        # 初始化MPS数据
        self._setup_mps_data()
        
    def _setup_camera_calibration(self):
        """设置相机标定参数"""
        rgb_camera_calibration = self.device_calibration.get_camera_calib(self.rgb_stream_label)
        
        self.rgb_linear_camera_calibration = calibration.get_linear_camera_calibration(
            int(rgb_camera_calibration.get_image_size()[0]),
            int(rgb_camera_calibration.get_image_size()[1]),
            rgb_camera_calibration.get_focal_lengths()[0],
            "pinhole",
            rgb_camera_calibration.get_transform_device_camera(),
        )
        
        self.rgb_rotated_linear_camera_calibration = calibration.rotate_camera_calib_cw90deg(
            self.rgb_linear_camera_calibration
        )
        self.camera_calibration = self.rgb_rotated_linear_camera_calibration
        
    def _setup_mps_data(self):
        """设置MPS数据"""
        mps_data_paths_provider = MpsDataPathsProvider(self.mps_folder)
        mps_data_paths = mps_data_paths_provider.get_data_paths()
        
        self.closed_loop_traj = mps.read_closed_loop_trajectory(
            str(mps_data_paths.slam.closed_loop_trajectory)
        )
        self.eyegaze_data = mps.read_eyegaze(mps_data_paths.eyegaze.general_eyegaze)
        self.wrist_and_palm_poses = mps.hand_tracking.read_wrist_and_palm_poses(
            mps_data_paths.hand_tracking.wrist_and_palm_poses
        )

    def _init_processing_threads(self):
        """初始化处理线程"""
        for _ in range(self.num_threads):
            thread = HandDetectionThread(
                self.input_queue,
                self.output_queue,
                self.model,
                self.model_cfg,
                self.detector,
                self.keypoint_detector,
                self.device,
                self.trajectory_buffer,
                self.stop_event
            )
            thread.daemon = True
            self.processing_threads.append(thread)

    def process_frame(self, data: SensorData, session_index: int, device_time_ns: int) -> None:
        """处理单帧数据"""
        if data.sensor_data_type() != SensorDataType.IMAGE:
            return
            
        # 获取图像数据
        img = data.image_data_and_record()[0].to_numpy_array()
        img = self.post_process_image(img)
        
        # 获取手部投影并更新轨迹缓冲区
        hand_projections = self.log_hand_tracking(device_time_ns)
        self.trajectory_buffer.update(hand_projections)
        
        # 创建处理任务
        output_dir = os.path.join(self.output_base_dir, f"frames_{session_index}")
        os.makedirs(output_dir, exist_ok=True)
        frame_index = len(os.listdir(output_dir))
        
        task = ProcessingTask(frame_index, img, device_time_ns, hand_projections)
        self.input_queue.put(task)

    def log_hand_tracking(
        self,
        device_time_ns: int,
        future_interval_seconds: float = 1.0,
        num_samples: int = 10
    ) -> str:
        """记录手部跟踪数据并生成轨迹预测"""
        if not self.wrist_and_palm_poses:
            return json.dumps({"left_hand": [], "right_hand": []}, indent=2)

        wrist_and_palm_pose = get_nearest_wrist_and_palm_pose(
            self.wrist_and_palm_poses, 
            device_time_ns
        )
        
        if wrist_and_palm_pose is None:
            return json.dumps({"left_hand": [], "right_hand": []}, indent=2)

        # 获取当前帧位姿
        current_pose_info = get_nearest_pose(self.closed_loop_traj, device_time_ns)
        if current_pose_info is None:
            return json.dumps({"left_hand": [], "right_hand": []}, indent=2)
            
        T_world_device_current = current_pose_info.transform_world_device
        
        left_hand_points = []
        right_hand_points = []

        # 处理左手数据
        if wrist_and_palm_pose.left_hand and wrist_and_palm_pose.left_hand.confidence >= 0:
            left_wrist_point_device = wrist_and_palm_pose.left_hand.wrist_position_device
            left_wrist_pixel = self.get_camera_projection_from_device_point(
                left_wrist_point_device, 
                self.camera_calibration
            )
            if left_wrist_pixel is not None:
                scaled_point = list(left_wrist_pixel)
                left_hand_points.append(scaled_point)

        # 处理右手数据
        if wrist_and_palm_pose.right_hand and wrist_and_palm_pose.right_hand.confidence > 0:
            right_wrist_point_device = wrist_and_palm_pose.right_hand.wrist_position_device
            right_wrist_pixel = self.get_camera_projection_from_device_point(
                right_wrist_point_device, 
                self.camera_calibration
            )
            if right_wrist_pixel is not None:
                scaled_point = list(right_wrist_pixel)
                right_hand_points.append(scaled_point)

        # 预测未来轨迹点
        future_interval_ns = int(future_interval_seconds * 1e9)
        future_timestamps = np.linspace(
            device_time_ns,
            device_time_ns + future_interval_ns,
            num_samples
        )

        for t_future in future_timestamps:
            future_wrist_and_palm_pose = get_nearest_wrist_and_palm_pose(
                self.wrist_and_palm_poses, 
                t_future
            )
            if future_wrist_and_palm_pose is None:
                continue

            # 获取未来帧的位姿
            future_pose_info = get_nearest_pose(self.closed_loop_traj, t_future)
            if future_pose_info is None:
                continue
                
            T_world_device_future = future_pose_info.transform_world_device

            # 预测左手未来位置
            if (future_wrist_and_palm_pose.left_hand and 
                future_wrist_and_palm_pose.left_hand.confidence > 0):
                future_left_wrist_point_device = future_wrist_and_palm_pose.left_hand.wrist_position_device
                future_left_wrist_point_world = T_world_device_future @ future_left_wrist_point_device
                future_left_wrist_point_device_first = T_world_device_current.inverse() @ future_left_wrist_point_world
                future_left_wrist_pixel = self.get_camera_projection_from_device_point(
                    future_left_wrist_point_device_first,
                    self.camera_calibration
                )
                if future_left_wrist_pixel is not None:
                    scaled_point = list(future_left_wrist_pixel)
                    left_hand_points.append(scaled_point)

            # 预测右手未来位置
            if (future_wrist_and_palm_pose.right_hand and 
                future_wrist_and_palm_pose.right_hand.confidence > 0):
                future_right_wrist_point_device = future_wrist_and_palm_pose.right_hand.wrist_position_device
                future_right_wrist_point_world = T_world_device_future @ future_right_wrist_point_device
                future_right_wrist_point_device_first = T_world_device_current.inverse() @ future_right_wrist_point_world
                future_right_wrist_pixel = self.get_camera_projection_from_device_point(
                    future_right_wrist_point_device_first,
                    self.camera_calibration
                )
                if future_right_wrist_pixel is not None:
                    scaled_point = list(future_right_wrist_pixel)
                    right_hand_points.append(scaled_point)

        result = {
            "left_hand": [
                {"wrist_point": point} for point in left_hand_points
            ] if len(left_hand_points) >= 2 else [],
            "right_hand": [
                {"wrist_point": point} for point in right_hand_points
            ] if len(right_hand_points) >= 2 else [],
        }

        return json.dumps(result, indent=2)

    def get_camera_projection_from_device_point(
        self, 
        point: np.ndarray, 
        camera_calibration
    ) -> Optional[np.ndarray]:
        """从设备坐标系获取相机投影点"""
        T_device_camera = camera_calibration.get_transform_device_camera()
        return camera_calibration.project(T_device_camera.inverse() @ point)

    def post_process_image(self, img: np.ndarray) -> np.ndarray:
        """后处理图像，包括畸变校正和旋转"""
        # 使用标定数据进行畸变校正
        img = calibration.distort_by_calibration(
            img,
            self.rgb_linear_camera_calibration,
            self.device_calibration.get_camera_calib(self.rgb_stream_label),
        )
        # 逆时针旋转90度
        img = np.rot90(img, k=3)
        return img

    def save_processed_frame(self, result: ProcessingResult, session_index: int):
        """保存处理后的帧"""
        output_dir = os.path.join(self.output_base_dir, f"frames_{session_index}")
        os.makedirs(output_dir, exist_ok=True)
        frame_path = os.path.join(output_dir, f"frame_{result.frame_id:05d}.png")
        print(f"Saving frame to {frame_path}")  # 添加日志
        cv2.imwrite(frame_path, result.processed_image)
        print(f"Successfully saved frame {result.frame_id}")  # 添加日志

    def result_handler(self, session_index: int):
        """处理结果的线程函数"""
        print("Starting result handler thread")  # 添加日志
        while not self.stop_event.is_set():
            try:
                result = self.output_queue.get(timeout=1.0)
                print(f"Got result for frame {result.frame_id}")  # 添加日志
                self.save_processed_frame(result, session_index)
                self.output_queue.task_done()
            except Empty:  # 从 queue 模块直接导入 Empty
                continue
            except Exception as e:
                print(f"Error in result handler: {e}")
                self.output_queue.task_done()

    def run(self):
        """运行手部跟踪系统"""
        print("Starting hand tracking system")  # 添加日志
        
        # 启动处理线程
        for thread in self.processing_threads:
            thread.start()
        
        # 启动结果处理线程
        result_thread = threading.Thread(target=self.result_handler, args=(1,))
        result_thread.daemon = True
        result_thread.start()
        
        # 设置数据传输选项
        deliver_option = self.provider.get_default_deliver_queued_options()
        deliver_option.deactivate_stream_all()
        deliver_option.activate_stream(self.rgb_stream_id)
        
        # 获取总帧数
        rgb_frame_count = self.provider.get_num_data(self.rgb_stream_id)
        
        try:
            # 处理每一帧
            with tqdm(total=rgb_frame_count) as progress_bar:
                for data in self.provider.deliver_queued_sensor_data(deliver_option):
                    device_time_ns = data.get_time_ns(TimeDomain.DEVICE_TIME)
                    self.process_frame(data, session_index=1, device_time_ns=device_time_ns)
                    progress_bar.update(1)
                    
                    # 控制生产速度，防止队列溢出
                    while self.input_queue.qsize() >= self.input_queue.maxsize * 0.9:
                        time.sleep(0.1)
        
        finally:
            print("Shutting down hand tracking system")  # 添加日志
            # 设置停止事件
            self.stop_event.set()
            
            # 等待所有任务完成
            print("Waiting for input queue to complete")  # 添加日志
            self.input_queue.join()
            print("Waiting for output queue to complete")  # 添加日志
            self.output_queue.join()
            
            # 等待所有线程结束
            print("Waiting for processing threads to complete")  # 添加日志
            for thread in self.processing_threads:
                thread.join()
            
            print("Waiting for result thread to complete")  # 添加日志
            result_thread.join()
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
            print("Hand tracking system shutdown complete")  # 添加日志


def main():
    vrs_path = "/data/borui/test/Cook-egg.vrs"
    mps_folder = "/data/borui/test/mps_Cook-egg_vrs"
    output_base_dir = "./output_frames"
    
    # 创建并运行手部跟踪系统
    tracking_system = HandTrackingSystem(vrs_path, mps_folder, output_base_dir, num_threads=2)
    tracking_system.run()

if __name__ == "__main__":
    main()