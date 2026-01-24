import os
import numpy as np
import yaml
from datetime import datetime
from typing import Optional, Dict, Any
from scipy.spatial.transform import Rotation

import sys
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))
from utils.setup_logger import setup_logger
from utils.utils import quat_to_euler

class DataRecorder:
    """
    专门用于记录和保存机器人episode数据的类
    每个数据类型保存为独立文件，每次保存创建新文件夹
    """
    
    def __init__(self, base_save_dir: str = "recordings", episode_id: Optional[str] = None):
        """
        初始化数据记录器
        
        Args:
            base_save_dir: 基础保存目录 (SpaceMan/recordings)
            episode_id: episode标识符，如果为None则自动生成时间戳
        """
        self.base_save_dir = base_save_dir
        self.episode_id = episode_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建本次保存的专用文件夹
        self.episode_dir = os.path.join(self.base_save_dir, f"episode_{self.episode_id}")
        os.makedirs(self.episode_dir, exist_ok=True)

        self.logger = setup_logger("DataRecorder")
        self.logger.info("Initializing DataRecorder")
        
        # 数据缓冲区
        self.actions = []
        self.robot_obs = []
        self.scene_obs = []
        self.rgb_static = []
        self.rgb_gripper = []
        
        # 计数器
        self.step_count = 0
        
    def initialize(self) -> None:
        self.clear_buffer()
        
        self.file_paths = {
            'actions': os.path.join(self.episode_dir, 'actions.npz'),
            'robot_obs': os.path.join(self.episode_dir, 'robot_obs.npz'),
            'scene_obs': os.path.join(self.episode_dir, 'scene_obs.npz'),
            'rgb_static': os.path.join(self.episode_dir, 'rgb_static.npz'),
            'rgb_gripper': os.path.join(self.episode_dir, 'rgb_gripper.npz')
        }
        
        self.logger.info(f"DataRecorder initialized")
        self.logger.info(f"Episode ID: {self.episode_id}")
        self.logger.info(f"Save directory: {self.episode_dir}")
        
    def record_actions(self, joint_positions, gripper) -> None:
        """
        记录单步actions数据
        
        Args:
            joint_positions: 关节位置 (6,)
            gripper: 夹爪值 (1,)
        """
        # 将关节位置和夹爪值组合成 (7,) 的action数组
        if joint_positions is None or gripper is None:
            raise ValueError("joint_positions and gripper cannot be None")
        
        joint_positions = np.asarray(joint_positions)
        gripper = np.asarray(gripper)
        
        # 确保形状正确
        if len(joint_positions.shape) == 0:
            joint_positions = np.array([joint_positions])
        if len(gripper.shape) == 0:
            gripper = np.array([gripper])
            
        # 组合成完整的action (7,)
        action = np.concatenate([joint_positions, gripper])
        assert action.shape == (7,), f"Action shape should be (7,), got {action.shape}"
        
        self.actions.append(action)
        self.step_count += 1
        
        self.logger.debug(f"Recorded actions: shape {action.shape}, step {self.step_count}")
        self.logger.debug(f"  Joint positions: {joint_positions}")
        self.logger.debug(f"  Gripper: {gripper}")

    def record_robot_obs(self, end_effector_pos, end_effector_quat, gripper_value, joint_positions, base_pos, base_quat) -> None:
        """
        记录单步robot_obs数据
        
        Args:
            end_effector_pos: 末端执行器位置 (3,)
            end_effector_quat: 末端执行器四元数 (4,)
            gripper_value: 夹爪值 (1,)
            joint_positions: 关节位置 (6,)
            base_pos: 基座位置 (3,)
            base_quat: 基座四元数 (4,)
        """
        # 转换为numpy数组
        end_effector_pos = np.asarray(end_effector_pos)
        end_effector_quat = np.asarray(end_effector_quat)
        gripper_value = np.asarray(gripper_value)
        joint_positions = np.asarray(joint_positions)
        base_pos = np.asarray(base_pos)
        base_quat = np.asarray(base_quat)
        
        # 确保标量转换为数组
        if len(gripper_value.shape) == 0:
            gripper_value = np.array([gripper_value])
        if len(joint_positions.shape) == 0:
            joint_positions = np.array([joint_positions])
            
        # 转换四元数为欧拉角
        ee_roll, ee_pitch, ee_yaw = quat_to_euler(end_effector_quat)
        base_roll, base_pitch, base_yaw = quat_to_euler(base_quat)
        
        # 组合成完整的robot_obs (19,)
        robot_obs = np.concatenate([
            end_effector_pos,                    # (3,) [x, y, z]
            [ee_roll, ee_pitch, ee_yaw],        # (3,) [roll, pitch, yaw]
            gripper_value,                      # (1,) [gripper]
            joint_positions,                    # (6,) [joint1, joint2, ..., joint6]
            base_pos,                           # (3,) [x, y, z]
            [base_roll, base_pitch, base_yaw]   # (3,) [roll, pitch, yaw]
        ])
        
        # 验证最终形状
        assert robot_obs.shape == (19,), f"Robot obs shape should be (19,), got {robot_obs.shape}"
        
        self.robot_obs.append(robot_obs)
        
        self.logger.debug(f"Recorded robot_obs: shape {robot_obs.shape}")
        self.logger.debug(f"  End effector pos: {end_effector_pos}")
        self.logger.debug(f"  End effector euler: [{ee_roll:.3f}, {ee_pitch:.3f}, {ee_yaw:.3f}]")
        self.logger.debug(f"  Gripper: {gripper_value}")
        self.logger.debug(f"  Joints: {joint_positions}")
        self.logger.debug(f"  Base: {base_pos}")
        self.logger.debug(f"  Base euler: [{base_roll:.3f}, {base_pitch:.3f}, {base_yaw:.3f}]")
        
    def record_scene_obs(self, position, quat) -> None:
        """
        记录单步scene_obs数据
        
        Args:
            position: 位置 (3,) - [x, y, z]
            quat: 四元数 (4,) - [qx, qy, qz, qw]
        """
        if position is None or quat is None:
            raise ValueError("position and quat cannot be None")
        
        position = np.asarray(position)
        quat = np.asarray(quat)
        
        # 确保形状正确
        if len(position.shape) == 0:
            position = np.array([position])
        if len(quat.shape) == 0:
            quat = np.array([quat])
            
        # 使用辅助函数转换四元数为欧拉角
        roll, pitch, yaw = quat_to_euler(quat)
        
        # 组合成完整的scene_obs (6,) - [x, y, z, roll, pitch, yaw]
        scene_obs = np.concatenate([position, [roll, pitch, yaw]])
        
        # 验证最终形状
        assert scene_obs.shape == (6,), f"Scene obs shape should be (6,), got {scene_obs.shape}"
        
        self.scene_obs.append(scene_obs)
        
        self.logger.debug(f"Recorded scene_obs: shape {scene_obs.shape}")
        self.logger.debug(f"  Position: {position}")
        self.logger.debug(f"  Euler angles: [{roll:.3f}, {pitch:.3f}, {yaw:.3f}]")
        
    def record_rgb_static(self, rgb_static: np.ndarray) -> None:
        """
        记录单步rgb_static数据
        
        Args:
            rgb_static: 静态摄像头RGB图像 (200, 200, 3)
        """
        rgb_static = np.asarray(rgb_static)
        assert rgb_static.shape == (200, 200, 3), f"RGB static shape should be (200, 200, 3), got {rgb_static.shape}"
        
        self.rgb_static.append(rgb_static)
        
        self.logger.debug(f"Recorded rgb_static: shape {rgb_static.shape}")
        
    def record_rgb_gripper(self, rgb_gripper: np.ndarray) -> None:
        """
        记录单步rgb_gripper数据
        
        Args:
            rgb_gripper: 夹爪摄像头RGB图像 (84, 84, 3)
        """
        rgb_gripper = np.asarray(rgb_gripper)
        assert rgb_gripper.shape == (84, 84, 3), f"RGB gripper shape should be (84, 84, 3), got {rgb_gripper.shape}"
        
        self.rgb_gripper.append(rgb_gripper)
        
        self.logger.debug(f"Recorded rgb_gripper: shape {rgb_gripper.shape}")
        
    def stop(self) -> Dict[str, str]:
        """
        停止记录并保存所有数据到五个独立的.npz文件
        
        Returns:
            Dict[str, str]: 各数据类型的保存文件路径
        """
        if self.step_count == 0:
            self.logger.warning("No data to save. Please record some steps first.")
            raise ValueError("No data to save. Please record some steps first.")
        
        self.logger.info(f"Stopping recording and saving episode {self.episode_id}")
        saved_files = {}

        if self.actions:
            actions_array = np.array(self.actions)  # (2771, 7)
            np.savez_compressed(self.file_paths['actions'],
                               data=actions_array,
                               episode_id=self.episode_id,
                               step_count=len(self.actions),
                               timestamp=datetime.now().isoformat())
            saved_files['actions'] = self.file_paths['actions']
            self.logger.info(f"Saved actions: {actions_array.shape} -> {self.file_paths['actions']}")

        if self.robot_obs:
            robot_obs_array = np.array(self.robot_obs)  # (2771, 19)
            np.savez_compressed(self.file_paths['robot_obs'],
                               data=robot_obs_array,
                               episode_id=self.episode_id,
                               step_count=len(self.robot_obs),
                               timestamp=datetime.now().isoformat())
            saved_files['robot_obs'] = self.file_paths['robot_obs']
            self.logger.info(f"Saved robot_obs: {robot_obs_array.shape} -> {self.file_paths['robot_obs']}")

        if self.scene_obs:
            scene_obs_array = np.array(self.scene_obs)  # (2771, 6)
            np.savez_compressed(self.file_paths['scene_obs'],
                               data=scene_obs_array,
                               episode_id=self.episode_id,
                               step_count=len(self.scene_obs),
                               timestamp=datetime.now().isoformat())
            saved_files['scene_obs'] = self.file_paths['scene_obs']
            self.logger.info(f"Saved scene_obs: {scene_obs_array.shape} -> {self.file_paths['scene_obs']}")
        
        if self.rgb_static:
            rgb_static_array = np.array(self.rgb_static)  # (2771, 200, 200, 3)
            np.savez_compressed(self.file_paths['rgb_static'],
                               data=rgb_static_array,
                               episode_id=self.episode_id,
                               step_count=len(self.rgb_static),
                               timestamp=datetime.now().isoformat())
            saved_files['rgb_static'] = self.file_paths['rgb_static']
            self.logger.info(f"Saved rgb_static: {rgb_static_array.shape} -> {self.file_paths['rgb_static']}")
        
        if self.rgb_gripper:
            rgb_gripper_array = np.array(self.rgb_gripper)  # (2771, 84, 84, 3)
            np.savez_compressed(self.file_paths['rgb_gripper'],
                               data=rgb_gripper_array,
                               episode_id=self.episode_id,
                               step_count=len(self.rgb_gripper),
                               timestamp=datetime.now().isoformat())
            saved_files['rgb_gripper'] = self.file_paths['rgb_gripper']
            self.logger.info(f"Saved rgb_gripper: {rgb_gripper_array.shape} -> {self.file_paths['rgb_gripper']}")
        
        self.logger.info(f"Episode {self.episode_id} saved successfully with {len(saved_files)} files")
        
        # # 生成YAML配置文件
        # config_file = self._generate_config_file()
        # if config_file:
        #     saved_files['config'] = config_file
            
        return saved_files
    
    def _generate_config_file(self) -> Optional[str]:
        """
        生成episode的YAML配置文件
        
        Returns:
            Optional[str]: 配置文件路径，如果生成失败则返回None
        """
        try:
            # 获取episode数据信息
            num_frames = len(self.actions) if self.actions else 0
            action_dim = self.actions[0].shape[0] if self.actions and len(self.actions) > 0 else 0
            robot_obs_dim = self.robot_obs[0].shape[0] if self.robot_obs and len(self.robot_obs) > 0 else 0
            scene_obs_dim = self.scene_obs[0].shape[0] if self.scene_obs and len(self.scene_obs) > 0 else 0
            
            self.logger.info(f"Generating config for episode {self.episode_id}: {num_frames} frames, "
                           f"action_dim={action_dim}, robot_obs_dim={robot_obs_dim}, scene_obs_dim={scene_obs_dim}")
            
            # 生成YAML配置
            config = {
                'cameras': {
                    'static': {
                        '_target_': 'vr_env.camera.static_camera.StaticCamera',
                        'name': 'static',
                        'fov': 10,
                        'aspect': 1,
                        'nearval': 0.01,
                        'farval': 10,
                        'width': 200,
                        'height': 200,
                        'look_at': [-0.026242351159453392, -0.0302329882979393, 0.3920000493526459],
                        'look_from': [2.871459009488717, -2.166602199425597, 2.555159848480571],
                        'up_vector': [0.4041403970338857, 0.22629790978217404, 0.8862616969685161]
                    },
                    'gripper': {
                        '_target_': 'vr_env.camera.gripper_camera.GripperCamera',
                        'name': 'gripper',
                        'fov': 75,
                        'aspect': 1,
                        'nearval': 0.01,
                        'farval': 2,
                        'width': 84,
                        'height': 84
                    },
                    'tactile': {
                        '_target_': 'vr_env.camera.tactile_sensor.TactileSensor',
                        'name': 'tactile',
                        'width': 120,
                        'height': 160,
                        'digit_link_ids': [10, 12],
                        'visualize_gui': False,
                        'config_path': 'conf/digit_sensor/config_digit.yml'
                    }
                },
                'load_dir': self.save_dir + '/',
                'data_path': 'data',
                'save_dir': None,
                'show_gui': False,
                'processes': 16,
                'max_episode_frames': num_frames,
                'save_body_infos': True,
                'set_static_cam': False,
                'env': {
                    'cameras': '${cameras}',
                    'show_gui': '${show_gui}',
                    'use_vr': False
                },
                'scene': {
                    '_target_': 'vr_env.scene.play_table_scene.PlayTableScene',
                    '_recursive_': False,
                    'name': 'robot_assisted_docking_scene',
                    'data_path': '${data_path}',
                    'global_scaling': 0.8,
                    'euler_obs': '${robot.euler_obs}',
                    'robot_base_position': [-0.34, -0.46, 0.24],
                    'robot_base_orientation': [0, 0, 0],
                    'robot_initial_joint_positions': [
                        -1.21779206, 1.03987646, 2.11978261, -2.34205014,
                        -0.87015947, 1.64119353, 0.55344866
                    ],
                    'surfaces': {
                        'table': [[0.0, -0.15, 0.46], [-0.35, -0.03, 0.46]],
                        'slider_left': [[-0.32, 0.05, 0.46], [-0.16, 0.12, 0.46]],
                        'slider_right': [[-0.05, 0.05, 0.46], [0.13, 0.12, 0.46]]
                    },
                    'objects': {
                        'movable_objects': {
                            'starlink_satellite': {
                                'file': 'starlink/urdf/starlink.urdf',
                                'initial_pos': 'any',
                                'initial_orn': 'any'
                            },
                            'docking_target': {
                                'file': 'docking_target/urdf/docking_target.urdf',
                                'initial_pos': 'any',
                                'initial_orn': 'any'
                            }
                        }
                    }
                },
                'robot': {
                    '_target_': 'spaceman.robots.satellite_manipulator.SatelliteManipulator',
                    'name': 'franka_merge',
                    'urdf_path': 'assets/urdf/franka_merge/urdf/franka_merge.urdf',
                    'euler_obs': True,
                    'gripper_obs': True,
                    'joint_pos_obs': True,
                    'tcp_pos_obs': True,
                    'tcp_quat_obs': True,
                    'base_pos_obs': True,
                    'base_quat_obs': True,
                    'gripper_value_obs': True
                },
                # 添加episode元数据
                'episode_metadata': {
                    'episode_id': self.episode_id,
                    'num_frames': num_frames,
                    'action_dim': action_dim,
                    'robot_obs_dim': robot_obs_dim,
                    'scene_obs_dim': scene_obs_dim,
                    'rgb_static_shape': (num_frames, 200, 200, 3) if self.rgb_static else None,
                    'rgb_gripper_shape': (num_frames, 84, 84, 3) if self.rgb_gripper else None,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            # 保存配置文件
            config_path = os.path.join(self.save_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
            
            self.logger.info(f"Generated config file: {config_path}")
            return config_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate config file: {e}")
            return None
    
    def load_data(self, data_type: str, episode_id: Optional[str] = None) -> np.ndarray:
        """
        加载特定类型的数据
        
        Args:
            data_type: 数据类型 ('actions', 'robot_obs', 'scene_obs', 'rgb_static', 'rgb_gripper')
            episode_id: episode ID，如果为None则使用当前episode
            
        Returns:
            np.ndarray: 加载的数据
        """
        if episode_id is None:
            episode_id = self.episode_id
            
        filepath = os.path.join(self.base_save_dir, f"episode_{episode_id}", f"{data_type}.npz")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        data = np.load(filepath)
        loaded_data = data['data']
        
        self.logger.info(f"Loaded {data_type} from episode {episode_id}: shape {loaded_data.shape}")
        return loaded_data
    
    def load_episode(self, episode_id: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        加载完整的episode数据
        
        Args:
            episode_id: episode ID，如果为None则使用当前episode
            
        Returns:
            Dict[str, np.ndarray]: 包含所有数据的字典
        """
        if episode_id is None:
            episode_id = self.episode_id
            
        episode_dir = os.path.join(self.base_save_dir, f"episode_{episode_id}")
        
        if not os.path.exists(episode_dir):
            raise FileNotFoundError(f"Episode folder not found: {episode_dir}")
        
        result = {}
        data_types = ['actions', 'robot_obs', 'scene_obs', 'rgb_static', 'rgb_gripper']
        
        for data_type in data_types:
            filepath = os.path.join(episode_dir, f"{data_type}.npz")
            if os.path.exists(filepath):
                data = np.load(filepath)
                result[data_type] = data['data']
        
        self.logger.info(f"Loaded episode {episode_id} from: {episode_dir}")
        self.logger.info(f"Available data types: {list(result.keys())}")
        
        return result
    
    def get_buffer_info(self) -> Dict[str, Any]:
        """
        获取当前缓冲区信息
        
        Returns:
            Dict: 包含缓冲区状态的字典
        """
        info = {
            'episode_id': self.episode_id,
            'episode_dir': self.episode_dir,
            'step_count': self.step_count,
            'actions_count': len(self.actions),
            'robot_obs_count': len(self.robot_obs),
            'scene_obs_count': len(self.scene_obs),
            'rgb_static_count': len(self.rgb_static),
            'rgb_gripper_count': len(self.rgb_gripper),
        }
        
        if self.step_count > 0:
            info['current_shapes'] = {
                'actions': (self.step_count, 7),
                'robot_obs': (self.step_count, 19),
                'scene_obs': (self.step_count, 6)
            }
            
            if self.rgb_static:
                info['current_shapes']['rgb_static'] = (self.step_count, 200, 200, 3)
            if self.rgb_gripper:
                info['current_shapes']['rgb_gripper'] = (self.step_count, 84, 84, 3)
        
        return info
    
    def clear_buffer(self) -> None:
        """清空缓冲区"""
        self.actions.clear()
        self.robot_obs.clear()
        self.scene_obs.clear()
        self.rgb_static.clear()
        self.rgb_gripper.clear()
        self.step_count = 0
        self.logger.info("Buffer cleared.")
    
    def __del__(self):
        """析构函数，确保数据被保存"""
        if self.step_count > 0:
            self.logger.warning(f"{self.step_count} steps of unsaved data. Call stop() to save.")


# 使用示例
if __name__ == "__main__":
    print("=== DataRecorder 使用示例 ===")
    
    # 创建记录器
    recorder = DataRecorder()
    recorder.initialize()
    # 模拟在task的每个step中调用record
    print("\n模拟task运行，记录数据...")
    for i in range(5):
        # 模拟获取机器人数据
        joint_positions = np.random.randn(6)  # 6个关节位置
        gripper = np.random.randn()           # 1个夹爪值
        robot_obs = np.random.randn(14)
        position = np.random.randn(3)         # [x, y, z]
        quat = np.random.randn(4)             # [qx, qy, qz, qw]
        rgb_static = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        rgb_gripper = np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)
        
        # 模拟robot_obs的各个组件
        ee_pos = np.random.randn(3)
        ee_quat = np.random.randn(4)
        gripper_val = np.random.randn()
        joints = np.random.randn(6)
        base_pos = np.random.randn(3)
        base_quat = np.random.randn(4)
        
        # 在每个step中分别记录数据 - 这就是在task中的用法
        recorder.record_actions(joint_positions, gripper)
        recorder.record_robot_obs(ee_pos, ee_quat, gripper_val, joints, base_pos, base_quat)
        recorder.record_scene_obs(position, quat)
        recorder.record_rgb_static(rgb_static)
        recorder.record_rgb_gripper(rgb_gripper)
        
        print(f"Step {i+1}: 数据已记录")
    
    # 模拟task结束或视窗关闭时调用stop
    print("\n模拟task结束，保存所有数据...")
    try:
        saved_files = recorder.stop()
        print(f"保存成功！文件数: {len(saved_files)}")
        for data_type, filepath in saved_files.items():
            print(f"  {data_type}: {filepath}")
    except Exception as e:
        print(f"保存失败: {e}")
    
    print("\n=== 示例完成 ===")
