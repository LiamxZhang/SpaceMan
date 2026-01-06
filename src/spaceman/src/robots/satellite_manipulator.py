import sys
import torch
import logging

# Extension APIs
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from robots.manipulator import Manipulator
from envs.genesis_env import GenesisSim
from utils.twolink_state import TwoLinkState
from utils.utils import convert_dict_to_tensors,map_to_range
from controllers.smooth_IK_solver import SmoothIKSolver

def setup_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        logger.addHandler(console_handler)
    
    return logger

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
        # self.logger.debug(f"Joints all: {self.joints_info}")
        
        self.logger.info("SatelliteManipulator robot initialization completed")

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
        self.logger.info(f"IK solver initialization completed")

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
            self.robot.set_qpos(qpos=joint_position, qs_idx_local=self.motors_qs)
            
            # Optional: Get feedback for verification
            qpos_fb = self.robot.get_qpos()
            self.logger.info(f"Current joint feedback position: {qpos_fb}")
            self.logger.info(f"Target joint position: {joint_position}")
            self.logger.info(f"joint index: {self.motors_qs}")
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

            gripper_value_current = self.get_gripper_value()

            self.logger.info(f"Gripper current value={gripper_value_current}")
            # self.logger.info(f"Gripper is set to gripper_value={gripper_value}")
            
            # Determine target finger state
            # finger_state = self.finger_open if gripper_open else self.finger_close
            finger_state = torch.tensor(gripper_value, dtype=self.datatype, device=self.device).expand(6)*self.config_gripper_joints
            self.logger.debug(finger_state)
            # finger_force = np.array([1.0, 1.0]) if gripper_open else np.array([-1.0, -1.0])
            # Control the gripper
            # self.robot.control_dofs_position(finger_state, self.fingers_qs)
            # self.robot.control_dofs_force(finger_force, self.fingers_qs)
            self.robot.set_qpos(qpos=finger_state, qs_idx_local=self.fingers_qs)
            
            # Update current state
            # self.gripper_state = gripper_value
            self.logger.info(f"Gripper control completed")
            return True
            
        except Exception as e:
            # Log the error
            error_msg = f"Failed to control gripper: {e}"
            self.logger.error(error_msg)
            return False