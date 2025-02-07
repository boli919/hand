import cv2
import numpy as np
import json
from typing import Optional, List, Tuple
import torch
from pathlib import Path
from tqdm import tqdm
import os
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

class HandTrackingSystem:
    def __init__(self, vrs_path: str, mps_folder: str, output_base_dir: str):
        """
        初始化手部跟踪系统
        
        Args:
            vrs_path: VRS文件路径
            mps_folder: MPS文件夹路径
            output_base_dir: 输出目录路径
        """
        self.vrs_path = vrs_path
        self.mps_folder = mps_folder
        self.output_base_dir = output_base_dir
        self.openpose_indices = list(range(21))
        self.gt_indices = self.openpose_indices
        
        # 初始化设备
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_keypoint = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型和配置
        self._init_models()
        self._init_providers()
        
    def _init_models(self):
        """初始化所有需要的模型和配置"""
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
        
    def get_camera_projection_from_device_point(
        self, 
        point: np.ndarray, 
        camera_calibration
    ) -> Optional[np.ndarray]:
        """从设备坐标系获取相机投影点"""
        T_device_camera = camera_calibration.get_transform_device_camera()
        return camera_calibration.project(T_device_camera.inverse() @ point)

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

    def detect_and_draw_hands(
        self,
        img: np.ndarray,
        hand_projections: Optional[str] = None,
    ) -> np.ndarray:
        """检测和绘制手部关键点"""
        img_copy = img.copy()
        left_wrist_point = None
        right_wrist_point = None
        
        # 解析手部投影数据
        if hand_projections:
            try:
                hand_data = json.loads(hand_projections)
                
                # 处理左手轨迹
                left_hand_points = hand_data.get("left_hand", [])
                if left_hand_points:
                    left_wrist_point = left_hand_points[0]["wrist_point"]
                    # 绘制轨迹点和线
                    points = []
                    for point in left_hand_points:
                        wrist_point = point["wrist_point"]
                        x, y = int(wrist_point[0]), int(wrist_point[1])
                        points.append((x, y))
                        cv2.circle(img_copy, (x, y), 2, (255, 255, 0), -1)
                    
                    # 使用曲线拟合绘制平滑轨迹
                    if len(points) >= 2:
                        points = np.array(points, dtype=np.int32)
                        # 使用 cv2.polylines 绘制平滑曲线
                        cv2.polylines(img_copy, [points], False, (255, 255, 0), 2, cv2.LINE_AA)
                
                # 处理右手轨迹
                right_hand_points = hand_data.get("right_hand", [])
                if right_hand_points:
                    right_wrist_point = right_hand_points[0]["wrist_point"]
                    # 绘制轨迹点和线
                    points = []
                    for point in right_hand_points:
                        wrist_point = point["wrist_point"]
                        x, y = int(wrist_point[0]), int(wrist_point[1])
                        points.append((x, y))
                        cv2.circle(img_copy, (x, y), 2, (255, 255, 0), -1)
                    
                    # 使用曲线拟合绘制平滑轨迹
                    if len(points) >= 2:
                        points = np.array(points, dtype=np.int32)
                        # 使用 cv2.polylines 绘制平滑曲线
                        cv2.polylines(img_copy, [points], False, (255, 255, 0), 2, cv2.LINE_AA)
                                
            except Exception as e:
                print(f"Error processing hand projections: {e}")
        
        img_cv2 = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
        
        # 使用检测器进行手部检测
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
            print("No valid bounding boxes detected. Returning original image.")
            return img_copy

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
                    pred_joints[0, 0] = right_wrist_point[0]  # x坐标
                    pred_joints[0, 1] = right_wrist_point[1]  # y坐标
                    
                elif not is_right and left_wrist_point is not None:
                    pred_joints[0, 0] = left_wrist_point[0]  # x坐标
                    pred_joints[0, 1] = left_wrist_point[1]  # y坐标
                
                v = np.ones((21, 1))
                pred_joints = np.concatenate((pred_joints, v), axis=-1)
                verts[:,0] = (2*is_right-1)*verts[:,0]
                cam_t = pred_cam_t_full[n]
                all_pred_2d.append(pred_joints)
                
        all_pred_2d = np.stack(all_pred_2d)

        # 添加判断条件，舍弃第一行数据差值不超过5的点
        to_remove = set()
        for i in range(len(all_pred_2d)):
            for j in range(i + 1, len(all_pred_2d)):
                if np.abs(all_pred_2d[i, 0, 0] - all_pred_2d[j, 0, 0]) <= 5:
                    to_remove.add(j)

        all_pred_2d = np.delete(all_pred_2d, list(to_remove), axis=0)

        # 绘制关键点
        input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
        input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2)
        pred_img = input_img.copy()[:,:,:-1][:,:,::-1] * 255
        
        for i in range(len(all_pred_2d)):
            body_keypoints_2d = all_pred_2d[i, :21].copy()
            for op, gt in zip(self.openpose_indices, self.gt_indices):
                if all_pred_2d[i, gt, -1] > body_keypoints_2d[op, -1]:
                    body_keypoints_2d[op] = all_pred_2d[i, gt]
            pred_img = render_openpose(pred_img, body_keypoints_2d)
        
        output_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
        
        return output_img

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

    def process_frame(
        self,
        data: SensorData,
        session_index: int,
        device_time_ns: int
    ) -> None:
        """处理单帧数据"""
        if data.sensor_data_type() != SensorDataType.IMAGE:
            return
            
        # 获取图像数据
        img = data.image_data_and_record()[0].to_numpy_array()
        
        # 应用图像后处理（畸变校正和旋转）
        img = self.post_process_image(img)
        
        # 获取手部投影
        hand_projections = self.log_hand_tracking(device_time_ns)
        
        # 检测和绘制手部
        processed_img = self.detect_and_draw_hands(img, hand_projections)
        
        # 保存处理后的图像
        output_dir = os.path.join(self.output_base_dir, f"frames_{session_index}")
        os.makedirs(output_dir, exist_ok=True)
        frame_index = len(os.listdir(output_dir))
        frame_path = os.path.join(output_dir, f"frame_{frame_index:05d}.png")
        cv2.imwrite(frame_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
        # 清理GPU缓存
        torch.cuda.empty_cache()

    def run(self):
        """运行手部跟踪系统"""
        # 设置数据传输选项
        deliver_option = self.provider.get_default_deliver_queued_options()
        deliver_option.deactivate_stream_all()
        deliver_option.activate_stream(self.rgb_stream_id)
        
        # 获取总帧数
        rgb_frame_count = self.provider.get_num_data(self.rgb_stream_id)
        
        # 处理每一帧
        with tqdm(total=rgb_frame_count) as progress_bar:
            for data in self.provider.deliver_queued_sensor_data(deliver_option):
                device_time_ns = data.get_time_ns(TimeDomain.DEVICE_TIME)
                self.process_frame(data, session_index=1, device_time_ns=device_time_ns)
                progress_bar.update(1)

def main():
    vrs_path = "/data/borui/test/Cook-egg.vrs"
    mps_folder = "/data/borui/test/mps_Cook-egg_vrs"
    output_base_dir = "./output_frames"
    
    # 创建并运行手部跟踪系统
    tracking_system = HandTrackingSystem(vrs_path, mps_folder, output_base_dir)
    tracking_system.run()

if __name__ == "__main__":
    main()