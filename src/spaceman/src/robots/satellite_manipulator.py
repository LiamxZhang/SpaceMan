import sys
import torch
import logging

# Extension APIs
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from robots.manipulator import Manipulator
from utils.utils import convert_dict_to_tensors,map_to_range
from controllers.smooth_IK_solver import SmoothIKSolver
from utils.setup_logger import setup_logger

class SatelliteManipulator(Manipulator):
    def __init__(
        self,
        name="franka_merge",
        sensors=[],
        backends=[]
    ):
        super().__init__(
            name=name,
            sensors=sensors,
            backends=backends
        )
        self.logger = setup_logger(f"SatelliteManipulator.{name}")
        
        # Logging the robot information
        self.logger.info(f"Joint dof indexes: {self.motors_dof+self.fingers_dof}")
        self.logger.info(f"Joint qs indexes: {self.motors_qs+self.fingers_qs}")
        # self.logger.info(f"Joints all: {self.joints_info}")
        

    def initialize(self):
        """After the scene is built."""
        # Initialize Franka robot's configuration
        self.set_config(self.config)

        # Initialize base and end effector's position and attitude
        self.update_state()
        self.logger.debug(f"End effector state: {self.ee_state}")

        self.IK = SmoothIKSolver(
            robot=self.robot, 
            end_effector=self.end_effector,
            smooth_factor=self.params["ik_params"]["smooth_factor"], 
            max_joint_change=self.params["ik_params"]["max_joint_change"],
        )
        self.logger.info(f"SatelliteManipulator robot {self._robot_name} initialization completed")


    def control_joint_pos(self, joint_position):
        """
        Input: 
        position: [x, y, z] (torch.tensor, list, tuple or numpy array)
        quaternion: [qw, qx, qy, qz] (torch.tensor, list, tuple or numpy array)
        """
        # Convert inputs to torch tensors with correct dtype and device if needed
        if not isinstance(joint_position, torch.Tensor):
            joint_position = torch.tensor(joint_position, dtype=self.datatype, device=self.device)
        else:
            joint_position = joint_position.to(dtype=self.datatype, device=self.device)
        
        # Ensure correct shape
        if joint_position.dim() == 0:
            joint_position = joint_position.unsqueeze(0)

        
        # Ensure joint position has shape [6]
        if joint_position.shape[-1] != 6:
            error_msg = f"Joint_position should have shape [6], got {joint_position.shape}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)       

        # Control joints' dofs
        try:
            # self.robot.set_dofs_position(position=joint_position, dofs_idx_local=self.motors_dof, zero_velocity=False) # 就是这个robot.set_qpos的问题
            # self.robot.set_qpos(qpos=joint_position, qs_idx_local=self.motors_qs, zero_velocity=False)
            self.robot.control_dofs_position(position=joint_position, dofs_idx_local=self.motors_dof) # control必须加物理引擎和刚度阻尼参数
            self._scene.step()

            # Optional: Get feedback for verification
            qpos_fb = self.robot.get_qpos()
            self.logger.debug(f"Current joint feedback position: {qpos_fb}")
            self.logger.debug(f"Target joint position: {joint_position}")
            self.logger.debug(f"joint index: {self.motors_qs}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to set joint positions: {e}")
            return False

    def control_gripper(self, gripper_value):
        """
        Control the gripper to open or close.
        Args:
            gripper_value: in range [0, 1]
                        1 to open gripper, 0 to close
        """
        try:         
            gripper_value = map_to_range(gripper_value, 0, 1, self.params["finger_close"][0], self.params["finger_open"][0])

            # self.logger.info(f"Gripper is set to gripper_value={gripper_value}")
            
            # Determine target finger state
            finger_state = torch.tensor(gripper_value, dtype=self.datatype, device=self.device).expand(6)*self.config_gripper_joints
            self.logger.debug(finger_state)

            # Control the gripper
            self.robot.control_dofs_position(position=finger_state, dofs_idx_local=self.fingers_dof)
            self._scene.step()
            # self.robot.set_qpos(qpos=finger_state, qs_idx_local=self.fingers_qs, zero_velocity=False)
            # self.robot.set_dofs_position(joint_position=self.finger_state, dofs_idx_local=self.fingers_dof, zero_velocity=False)  #还有它的问题
            
            _ = self.get_gripper_value()
            self.logger.debug(f"\rGripper current value={self.gripper_value}")
            
            if self.gripper_value > self.params["finger_open"][0] and self.gripper_state == 1.0:
                self.gripper_state = 0.0
            elif self.gripper_value < self.params["finger_close"][0] and self.gripper_state == 0.0:
                self.gripper_state = 1.0
            # Update current state
            # self.gripper_state = gripper_value
            self.logger.debug(f"\rGripper control completed")
            return True
            
        except Exception as e:
            # Log the error
            error_msg = f"Failed to control gripper: {e}"
            self.logger.error(error_msg)
            return False