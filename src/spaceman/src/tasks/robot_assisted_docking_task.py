"""RobotAssistedDockingTask: Combined simulation of a Starlink satellite bus and its onboard manipulator.

This task runs inside the GenesisSim environment (no ROS2 dependency) and:
- Adds a Starlink satellite URDF model to the scene.
- Spawns a `SatelliteManipulator` robot (same as used in the ROS node).
- Steps both simulation and robot each call to `step()`.
- Tracks simple episode statistics/reward.
"""

import sys
import time
import torch
import genesis as gs
from typing import Optional

# Extension APIs
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))
from envs.genesis_env import GenesisSim
from tasks.task import Task, TaskStatus
from robots.robot import Robot
from robots.satellite_manipulator import SatelliteManipulator
from configs.asset_configs import get_asset, get_configs, get_pid
from utils.utils import map_to_range

class RobotAssistedDockingTask(Task):
    def __init__(self, config: Optional[dict] = None):
        super().__init__(task_name="robot_assisted_docking")
        # Optional external config (not heavily used right now)
        self.user_config = config or {}
        # Hold references to spawned entities
        self.starlink: Optional[Robot] = None
        self.starlink_manipulator: Optional[SatelliteManipulator] = None
        
        # Control interface attributes
        self.joint_positions = None
        self.gripper_value = 1.0
        self.current_setpoint = [1, 0, 0, -2, 0, 0]  # Default PID setpoint

    # ------------------------------------------------------------------
    # Task lifecycle methods
    # ------------------------------------------------------------------
    def initialize(self) -> bool:
        """Spawn starlink model + manipulator and start simulation."""
        try:
            # Add starlink asset to the scene
            if self.starlink is None:
                self.starlink = Robot(name="starlink")
                
            # Create starlink_manipulator robot
            if self.starlink_manipulator is None:
                self.starlink_manipulator = SatelliteManipulator(name="franka_merge", sensors=[], backends=[])

            # Start viewer/simulation if not running
            self.gsim.start()
            self.starlink.initialize()
            self.starlink_manipulator.initialize()

            self.status = TaskStatus.RUNNING
            self.logger.info(f"RobotAssistedDockingTask initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.status = TaskStatus.FAILED
            return False

    def step(self, dt: Optional[float] = None) -> bool:
        """Advance simulation one tick."""
        if self.status != TaskStatus.RUNNING:
            return False
        try:
            # Step entities
            if self.starlink_manipulator:
                self.starlink_manipulator.step()
            if self.starlink:
                self.starlink.step()

            self.gsim.step()
            self.step_count += 1

            # Compute total reward
            self.total_reward += self.reward().item() if isinstance(self.reward(), torch.Tensor) else self.reward()
            
            # Terminate condition example
            if self.check_termination():
                self.status = TaskStatus.COMPLETED
            return True
        except Exception as e:
            self.logger.error(f"Step error: {e}")
            self.status = TaskStatus.FAILED
            return False

    def control(self, joints: list = None, pid_setpoint: list = None) -> bool:
        """
        Control interface for starlink_manipulator using ROS-style joint state message.
        
        Args:
            joint_msg: ROS joint state message with name and position fields
            gripper_value: Optional gripper control value (0.0=open, 1.0=close)
            pid_setpoint: Optional PID setpoint for joints [6 values]
            
        Returns:
            bool: True if control successful, False otherwise
        """
        try:
            if not self.starlink_manipulator or self.status != TaskStatus.RUNNING:
                self.logger.warning("Manipulator not available or task not running")
                return False
            
            # Process joint positions
            self.joint_positions = joints  # Take first 6 joints,已经切了前六个，且已经校准方向
            
            gripper_value = joints[6]
            self.gripper_value = map_to_range(gripper_value, 0.0, 1.0, 0.0, 1.0)
            
            # Update PID setpoint if provided
            if pid_setpoint is not None:
                self.current_setpoint = pid_setpoint # 传list即可，pid内部update会转换
                self._update_pid_setpoint()
            
            # Apply controls
            joint_tensor = torch.tensor(self.joint_positions, dtype=self.datatype, device=self.device)
            self.starlink_manipulator.control_gripper(self.gripper_value)
            self.starlink_manipulator.control_joint_pos(joint_tensor)
            
            self.logger.debug(f"Control applied: joints={self.joint_positions}, gripper={self.gripper_value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Control error: {e}")
            return False
    
    def _update_pid_setpoint(self) -> bool:
        """
        Update PID controller setpoint for the manipulator using the existing update_setpoint method.
        
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            if hasattr(self.starlink_manipulator, 'pid'):
                self.starlink_manipulator.pid.update_setpoint(self.current_setpoint)
                self.logger.debug(f"PID setpoint updated: {self.current_setpoint}")
                return True
            else:
                self.logger.warning("Manipulator has no PID controller")
                return False
        except Exception as e:
            self.logger.error(f"PID setpoint update error: {e}")
            return False

    def stop(self) -> bool:
        """Stop task execution and clean up resources."""
        try:
            if self.status not in [TaskStatus.RUNNING, TaskStatus.PAUSED]:
                return False
                
            # Stop simulation
            if hasattr(self, 'gsim') and self.gsim:
                self.gsim.stop()
            
            # Clean up robot resources
            if self.starlink_manipulator:
                try:
                    if hasattr(self.starlink_manipulator, 'stop'):
                        self.starlink_manipulator.stop()
                    else:
                        self.logger.warning("SatelliteManipulator has no stop method")
                except Exception as e:
                    self.logger.warning(f"Error stopping manipulator: {e}")
            
            if self.starlink:
                try:
                    if hasattr(self.starlink, 'stop'):
                        self.starlink.stop()
                    else:
                        self.logger.warning("Robot has no stop method")
                except Exception as e:
                    self.logger.warning(f"Error stopping starlink: {e}")
            
            # Update status
            self.status = TaskStatus.CANCELLED
            self.logger.info(f"RobotAssistedDockingTask stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during stop: {e}")
            self.status = TaskStatus.FAILED
            return False

    def reset(self) -> bool:
        """Reset the task by reloading the scene."""
        try:
            # Stop current sim and create a fresh GenesisSim scene
            self.gsim.stop()
            self.gsim = GenesisSim()
            self.starlink = None
            self.starlink_manipulator = None
            self.step_count = 0
            self.total_reward = 0.0
            self.success = False
            self.status = TaskStatus.NOT_STARTED
            return self.initialize()
        except Exception as e:
            self.logger.error(f"Reset failed: {e}")
            self.status = TaskStatus.FAILED
            return False

    def reward(self) -> torch.Tensor:
        """Very simple reward: always zero (placeholder)."""
        return torch.tensor(0.0, dtype=self.datatype, device=self.device)


def main():
    """Test RobotAssistedDockingTask: initialize, run a few steps, then clean up."""
    print("=== RobotAssistedDockingTask Test ===")
    
    task = RobotAssistedDockingTask()
    
    # Initialize the task
    if not task.initialize():
        print("❌ Task initialization failed")
        return
    
    print("✅ Task initialized successfully")
    
    # Create a mock joint state message for testing
    class MockJointMsg:
        def __init__(self):
            self.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
            self.position = [0.1, 0.2, -0.1, 0.5, -0.3, 0.0, 0.5]
    
    # Test control function
    try:
        mock_msg = MockJointMsg()
        
        print("🎮 Testing control function...")
        
        # Test 1: Basic joint control
        success = task.control(mock_msg)
        print(f"📊 Basic control: {'✅' if success else '❌'}")
        
        # Test 2: Joint control with gripper
        success = task.control(mock_msg, gripper_value=0.8)
        print(f"🦾 Control with gripper: {'✅' if success else '❌'}")
        
        # Test 3: Joint control with gripper and PID setpoint
        success = task.control(mock_msg, gripper_value=0.3, pid_setpoint=[0.5, 0.5, 0.5, -1.0, 0.0, 0.0])
        print(f"⚙️ Control with PID setpoint: {'✅' if success else '❌'}")
        
        # Test 4: Invalid task status
        task.status = TaskStatus.COMPLETED
        success = task.control(mock_msg)
        print(f"🚫 Control when not running: {'✅' if not success else '❌'}")
        
        # Reset status for further tests
        task.status = TaskStatus.RUNNING
        
        print("🎯 All control tests completed!")
        
    except Exception as e:
        print(f"❌ Control test error: {e}")
    
    # Run a few simulation steps
    print("🔄 Running simulation steps...")
    try:
        for i in range(3):
            success = task.step()
            print(f"Step {i+1}: {'✅' if success else '❌'}")
            if not success:
                break
    except Exception as e:
        print(f"❌ Step error: {e}")
    
    # Clean up
    print("🧹 Cleaning up...")
    # task.stop()
    print("✅ Test completed!")
    print(f"Status: {task.status}")
    print(f"Step count: {task.step_count}")
    
    # Run simulation for a few steps
    max_steps = 50
    print(f"\nRunning simulation for {max_steps} steps...")
    
    try:
        for i in range(max_steps):
            if not task.step():
                print(f"❌ Step {i} failed")
                break
            
            if i % 10 == 0:
                print(f"Step {i}: status={task.status}, reward={task.reward().item():.3f}")
            
            # Small delay to make simulation observable
            time.sleep(0.1)
            
            # Check if task completed
            if task.status == TaskStatus.COMPLETED:
                print(f"✅ Task completed at step {i}")
                break
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
    finally:
        print(f"\nFinal stats:")
        print(f"- Status: {task.status}")
        print(f"- Steps: {task.step_count}")
        print(f"- Total reward: {task.total_reward:.3f}")
        
        # Clean up
        try:
            task.reset()
            print("✅ Task reset successfully")
        except Exception as e:
            print(f"⚠️  Reset failed: {e}")
    
    print("=== Test Complete ===")


if __name__ == "__main__":
    main()