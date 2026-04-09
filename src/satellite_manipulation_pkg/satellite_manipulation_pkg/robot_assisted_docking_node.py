#!/usr/bin/env python3
"""
ROS2 Node for Robot Assisted Docking Task
This node integrates the RobotAssistedDockingTask with ROS2 communication
"""

import sys
import copy
import numpy as np
import torch
import threading
import pygame
from pygame.locals import *

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import Header, Float64

# Extension APIs
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
workspace_src_path = current_file_path.parent.parent
spaceman_path = workspace_src_path.joinpath("spaceman/src")
sys.path.append(str(spaceman_path))

from tasks.robot_assisted_docking_task import RobotAssistedDockingTask
from utils.utils import map_to_range


class RobotAssistedDockingNode(Node):
    """ROS2 Node for Robot Assisted Docking Task"""
    
    def __init__(self):
        super().__init__('robot_assisted_docking_node')
        
        # Initialize the docking task
        self.task = None
        self.initialize()

        # Initialize keyboard control
        self.keyboard_control_enabled = False
        # self.init_keyboard_control()
        
        # ROS2 Publishers
        self.ee_pose_pub = self.create_publisher(
            PoseStamped,
            '/ee_pose_broadcaster/global_pose',
            10
        )

        # create gripper state publisher
        self.gripper_state_pub = self.create_publisher(
            Float64,
            '/ee_pose_broadcaster/gripper_state',
            10
        )

        # Subscribe to keyboard control messages
        self.keyboard_setpoint_sub = self.create_subscription(
            PoseStamped,
            '/keyboard_control/signal',
            self.keyboard_setpoint_callback,
            10)

        # ROS2 Subscribers
        self.joint_command_sub = self.create_subscription(
            JointState,
            '/wx250s/joint_states',
            self.joint_state_callback,
            10)

        # Timer for simulation updates
        self.timer = self.create_timer(0.01, self.timer_callback)  # 100Hz
        
        # State variables
        self.is_running = True
        self.gripper_value = 1.0
        self.gripper_state = True
        self.current_setpoint = None
        self.get_logger().info("Robot Assisted Docking Node initialized")
        
    def initialize(self):
        """Initialize the RobotAssistedDockingTask"""
        try:
            self.task = RobotAssistedDockingTask()
            self.task.initialize()
            self.get_logger().info("RobotAssistedDockingTask initialized successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize task: {e}")
            raise

    def main_thread(self):
        """RobotAssistedDockingTask main loop"""
        self.counter = 0.0
        self.rate = self.create_rate(10, self.get_clock())  # 设置循环频率为10Hz
        while self.is_running:
            try:
                # Step the task
                done = self.task.step()
                
                # Update and publish state
                self.end_effector_update()
                self.gripper_update()
                
                # Check if task is done
                if done:
                    self.get_logger().info("Task completed!")
                    self.task.reset()
                
                self.counter += 1
                self.rate.sleep()
                
            except Exception as e:
                self.get_logger().error(f"Error in main thread: {e}")
                break
        
        # Shutdown
        self.get_logger().info("Main thread stopped.")
        if self.task:
            self.task.stop()
    
    def timer_callback(self):
        """模拟步进函数，由定时器触发"""
        if not self.is_running or self.task is None:
            self.get_logger().info("Viewer window has been closed.")
            self.timer.cancel()
            self.task.stop()
            return
        # main
        try:
            # Step the task (returns bool for success)
            success = self.task.step()
            
            if success:
                # Update and publish state
                self.end_effector_update()
                self.gripper_update()
            
                # Check if task is completed
                if self.task.status.value == "completed":
                    self.get_logger().info("Task completed!")
                    self.task.reset()
            else:
                # Check if this is due to viewer closure (normal stop) vs actual error
                if self.task.status.value in ["failed", "stopped", "cancelled"]:
                    self.get_logger().info("Simulation stopped")
                    self.is_running = False
                    self.destroy_node()
                    return
                else:
                    self.get_logger().warning("Task step failed")
                
        except Exception as e:
            # Check if this is a viewer closure related exception
            if "viewer" in str(e).lower() or "window" in str(e).lower():
                self.get_logger().info("Viewer window closed, stopping simulation")
                self.is_running = False
                self.destroy_node()
                return
            else:
                self.get_logger().error(f"Error in timer callback: {e}")

    def end_effector_update(self):
        """
        将PyTorch tensor格式的位置和四元数转换为ROS2 PoseStamped消息并发布
        从RobotAssistedDockingTask获取机器人状态
        """
        try:
            # 从task获取机器人状态
            if hasattr(self.task, 'starlink_manipulator') and self.task.starlink_manipulator:
                position_tensor = self.task.starlink_manipulator.ee_global_position
                quat_tensor = self.task.starlink_manipulator.ee_global_quaternion
            else:
                self.get_logger().error("No manipulator available for EE pose update")
                return

            # 确保数据在CPU上并且是numpy数组格式
            if position_tensor.is_cuda:
                position_np = position_tensor.cpu().numpy()
            else:
                position_np = position_tensor.numpy()
            
            if quat_tensor.is_cuda:
                quat_np = quat_tensor.cpu().numpy()
            else:
                quat_np = quat_tensor.numpy()
            
            # 创建PoseStamped消息
            pose_msg = PoseStamped()
            
            # 设置header
            pose_msg.header = Header()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            
            # 设置位置 (x, y, z)
            pose_msg.pose.position.x = float(position_np[0])
            pose_msg.pose.position.y = float(position_np[1])
            pose_msg.pose.position.z = float(position_np[2])
            
            # 设置四元数 (注意ROS2使用[x, y, z, w]顺序)
            # 你的tensor是[qw, qx, qy, qz]，需要转换为[x, y, z, w]
            pose_msg.pose.orientation.w = float(quat_np[0])  # qw
            pose_msg.pose.orientation.x = float(quat_np[1])  # qx
            pose_msg.pose.orientation.y = float(quat_np[2])  # qy  
            pose_msg.pose.orientation.z = float(quat_np[3])  # qz
            
            # 发布消息
            self.ee_pose_pub.publish(pose_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error updating end-effector pose: {e}")
    
    def gripper_update(self):
        """从RobotAssistedDockingTask获取夹爪状态并发布"""
        try:
            # 从task获取夹爪状态 gripper_state是状态但是map_to_range后的alue
            if hasattr(self.task, 'starlink_manipulator') and self.task.starlink_manipulator:
                gripper_state = self.task.starlink_manipulator.gripper_state
            else:
                self.get_logger().error("No manipulator available for gripper update")
                return

            # hand open: True->1.0, hand close: False->0.0
            if hasattr(gripper_state, 'item'):  # 如果是 torch.tensor
                gripper_value = gripper_state.item()
            else:  # 如果是 Python bool
                gripper_value = gripper_state
        # print(gripper_state,gripper_value)
            # 创建Float64消息
            gripper_msg = Float64()
            gripper_msg.data = float(gripper_value)
            
            # 发布消息
            self.gripper_state_pub.publish(gripper_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error updating gripper state: {e}")
    

    def keyboard_setpoint_callback(self, msg):
        """
        接收来自keyboard_control的current_setpoint变化量
        直接更新current_setpoint
        """
        try:
            # 从PoseStamped消息中提取位置和方向
            position = msg.pose.position
            orientation = msg.pose.orientation
            
            # 转换为torch tensor
            position_tensor = torch.tensor([position.x, position.y, position.z], dtype=torch.float32)
            # 只使用qx, qy, qz的三维向量
            orientation_tensor = torch.tensor([orientation.x, orientation.y, orientation.z], dtype=torch.float32)
            
            # 合并为6维setpoint [x, y, z, qx, qy, qz]
            keyboard_setpoint = torch.cat([position_tensor, orientation_tensor])
            
            # 检查变化量是否均为零
            if torch.allclose(keyboard_setpoint, torch.zeros_like(keyboard_setpoint), atol=1e-6):
                # 如果变化量均为零，不更新setpoint
                self.current_setpoint = None
                self.get_logger().debug("Keyboard setpoint change is zero, not updating")
                return
            
            current_setpoint = self.task.starlink_manipulator.pid.setpoint
            # print("Current Setpoint = ",current_setpoint)
            self.current_setpoint = current_setpoint + keyboard_setpoint
            
            self.get_logger().debug(f"Updated setpoint from keyboard: {self.current_setpoint}")
            
        except Exception as e:
            self.get_logger().error(f"Error in keyboard_setpoint_callback: {e}")

    def joint_state_callback(self, msg):
        try:
            self.joint_names = msg.name
            self.msg_positions = msg.position
            self.joint_positions = self.msg_positions

            self.log_joint_states()
            self.get_gripper_state()
            
            # 使用task的控制接口
            if self.task:
                self.task.control(joints=self.joint_positions, pid_setpoint=self.current_setpoint)
            else:
                self.get_logger().error("No task available for control")
        except Exception as e:
            self.get_logger().error(f"Joint state callback error: {e}")

    def log_joint_states(self):
        if not hasattr(self, 'joint_names') or not self.joint_names:
            return
        info_lines = [f""]
        for i, name in enumerate(self.joint_names):
            line = f"  {name}: "
            if i < len(self.msg_positions):
                line += f"pos={self.msg_positions[i]:.3f}rad "
            info_lines.append(line)
        self.get_logger().info('\n'.join(info_lines))

    def get_gripper_state(self):
        try:
            gripper_value = self.msg_positions[6]
            self.gripper_value = map_to_range(gripper_value, 0.0, 1.0, 0.0, 1.0)
        except Exception as e:
            error_msg = f"Error updating gripper state: {e}"
            if hasattr(self, 'get_logger'):
                self.get_logger().error(error_msg)
            else:
                print(f"ERROR: {error_msg}")
        self.get_logger().info(f"gripper state = {'Open' if self.gripper_state else 'Close'}")

    def destroy_node(self):
        if hasattr(self, 'timer'):
            self.timer.cancel()
        if hasattr(self, 'task'):
            self.task.stop()
        super().destroy_node()

def main():
    rclpy.init()
    node = RobotAssistedDockingNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        sys.exit()

if __name__ == '__main__':
    main()
