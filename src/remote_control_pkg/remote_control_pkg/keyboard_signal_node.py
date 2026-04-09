#!/usr/bin/env python3
"""
ROS2 Node for Keyboard Signal Control
This node listens to WSAD keyboard input and publishes x,y directional changes
"""

import sys
import numpy as np
import torch
import pygame
from pygame.locals import *

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import Header

class KeyboardSignalNode(Node):
    """ROS2 Node for WSAD Keyboard Signal Control"""
    
    def __init__(self):
        super().__init__('keyboard_signal_node')
        
        # Create publisher for keyboard signals
        self.signal_pub = self.create_publisher(
            PoseStamped,
            '/keyboard_control/signal',
            10
        )
        
        # Initialize pygame for keyboard input
        pygame.init()
        self.screen = pygame.display.set_mode((200, 120))  # Increased height for additional info
        pygame.display.set_caption('WSAD Control')
        
        # Control parameters
        self.step_size = 0.01  # Step size for x,y movement
        self.current_signal = np.array([0.0, 0.0, 0.0])  # [x, y, z]
        
        # Total accumulated signal (for UI display)
        self.total_accumulated_signal = np.array([0.0, 0.0, 0.0])  # Total since init
        
        # Long press parameters
        self.long_press_enabled = True
        self.long_press_delay = 0.5  # seconds before long press starts
        self.long_press_interval = 0.1  # seconds between long press signals
        self.key_press_time = {}  # Track when each key was pressed
        self.last_long_press_time = {}  # Track last long press signal time
        
        # Keyboard state
        self.keys_pressed = {
            'w': False,  # x positive
            's': False,  # x negative  
            'a': False,  # y negative
            'd': False   # y positive
        }
        
        # Clock for pygame timing
        self.clock = pygame.time.Clock()
        
        self.get_logger().info("Keyboard Signal Node initialized")
        self.get_logger().info("Controls: W/S for X-axis, A/D for Y-axis")
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            elif event.type == KEYDOWN:
                self.handle_keydown(event.key)
            elif event.type == KEYUP:
                self.handle_keyup(event.key)
        return True
    
    def handle_keydown(self, key):
        """Handle key press events"""
        current_time = pygame.time.get_ticks() / 1000.0  # Convert to seconds
        
        if key == K_w:
            self.keys_pressed['w'] = True
            self.key_press_time['w'] = current_time
            self.last_long_press_time['w'] = current_time
        elif key == K_s:
            self.keys_pressed['s'] = True
            self.key_press_time['s'] = current_time
            self.last_long_press_time['s'] = current_time
        elif key == K_a:
            self.keys_pressed['a'] = True
            self.key_press_time['a'] = current_time
            self.last_long_press_time['a'] = current_time
        elif key == K_d:
            self.keys_pressed['d'] = True
            self.key_press_time['d'] = current_time
            self.last_long_press_time['d'] = current_time
    
    def handle_keyup(self, key):
        """Handle key release events"""
        if key == K_w:
            self.keys_pressed['w'] = False
            if 'w' in self.key_press_time:
                del self.key_press_time['w']
            if 'w' in self.last_long_press_time:
                del self.last_long_press_time['w']
        elif key == K_s:
            self.keys_pressed['s'] = False
            if 's' in self.key_press_time:
                del self.key_press_time['s']
            if 's' in self.last_long_press_time:
                del self.last_long_press_time['s']
        elif key == K_a:
            self.keys_pressed['a'] = False
            if 'a' in self.key_press_time:
                del self.key_press_time['a']
            if 'a' in self.last_long_press_time:
                del self.last_long_press_time['a']
        elif key == K_d:
            self.keys_pressed['d'] = False
            if 'd' in self.key_press_time:
                del self.key_press_time['d']
            if 'd' in self.last_long_press_time:
                del self.last_long_press_time['d']
    
    def update_signal(self):
        """Update current signal based on keyboard state and long press logic"""
        current_time = pygame.time.get_ticks() / 1000.0  # Convert to seconds
        
        # Reset current signal when no keys are pressed
        if not any(self.keys_pressed.values()):
            self.current_signal = np.array([0.0, 0.0, 0.0])
            return
        
        # Determine if we should publish a signal
        should_publish = False
        
        for key in ['w', 's', 'a', 'd']:
            if self.keys_pressed[key] and key in self.key_press_time:
                press_duration = current_time - self.key_press_time[key]
                
                if press_duration >= self.long_press_delay:
                    if current_time - self.last_long_press_time[key] >= self.long_press_interval:
                        self.last_long_press_time[key] = current_time
                        should_publish = True
                        break
                elif press_duration < 0.1:
                    should_publish = True
                    break
        
        # Generate signal to publish
        if should_publish:
            # Generate a fixed step signal
            signal_step = np.array([0.0, 0.0, 0.0])
            for key in ['w', 's', 'a', 'd']:
                if self.keys_pressed[key]:
                    if key == 'w':
                        signal_step[0] += self.step_size  # +x
                    elif key == 's':
                        signal_step[0] -= self.step_size  # -x
                    elif key == 'a':
                        signal_step[1] -= self.step_size  # -y
                    elif key == 'd':
                        signal_step[1] += self.step_size  # +y
            
            self.current_signal = signal_step
            
            # Update total accumulated signal for UI display
            self.total_accumulated_signal += signal_step
        else:
            self.current_signal = np.array([0.0, 0.0, 0.0])
    
    def get_display_signal(self):
        """Get the total accumulated signal for UI display"""
        return self.total_accumulated_signal.copy()
    
    def draw_ui(self):
        """Draw simple UI showing current state"""
        font = pygame.font.Font(None, 24)
        
        # Get accumulated signal for display
        display_signal = self.get_display_signal()
        
        # Show accumulated signal
        signal_text = f"X: {display_signal[0]:.3f}, Y: {display_signal[1]:.3f}"
        text_surface = font.render(signal_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))
        
        # Show actual signal being published (when non-zero)
        if np.any(self.current_signal != 0):
            actual_text = f"Send: {self.current_signal[0]:.3f}, {self.current_signal[1]:.3f}"
            text_surface = font.render(actual_text, True, (255, 255, 100))
            self.screen.blit(text_surface, (10, 70))
        
        # Show active keys and long press status
        current_time = pygame.time.get_ticks() / 1000.0
        active_keys = []
        long_press_keys = []
        
        for key in ['w', 's', 'a', 'd']:
            if self.keys_pressed[key] and key in self.key_press_time:
                press_duration = current_time - self.key_press_time[key]
                active_keys.append(key.upper())
                if press_duration >= self.long_press_delay:
                    long_press_keys.append(key.upper())
        
        if active_keys:
            if long_press_keys:
                keys_text = f"Keys: {', '.join(active_keys)} (LONG)"
                text_surface = font.render(keys_text, True, (255, 100, 100))
            else:
                keys_text = f"Keys: {', '.join(active_keys)}"
                text_surface = font.render(keys_text, True, (100, 255, 100))
            self.screen.blit(text_surface, (10, 40))
    
    def publish_signal(self):
        """Publish current keyboard signal as PoseStamped"""
        try:
            # Create PoseStamped message
            msg = PoseStamped()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "keyboard_signal_frame"
            
            # Set position (x, y changes)
            msg.pose.position = Point(
                x=float(self.current_signal[0]),
                y=float(self.current_signal[1]), 
                z=float(self.current_signal[2])
            )
            
            # Set orientation (qx, qy, qz - all zero for now)
            msg.pose.orientation = Quaternion(
                x=0.0,
                y=0.0,
                z=0.0,
                w=0.0  # Not used, but required
            )
            
            # Publish message
            self.signal_pub.publish(msg)
            
            # Debug logging
            if np.any(self.current_signal != 0):
                self.get_logger().debug(f"Published signal: X={self.current_signal[0]:.3f}, Y={self.current_signal[1]:.3f}")
            
        except Exception as e:
            self.get_logger().error(f"Error publishing signal: {e}")
    
    def run(self):
        """Main run loop"""
        try:
            while rclpy.ok():
                # Handle pygame events
                if not self.handle_events():
                    break
                
                # Update signal based on keyboard state
                self.update_signal()
                
                # Publish signal
                self.publish_signal()
                
                # Draw UI
                self.screen.fill((50, 50, 50))
                self.draw_ui()
                pygame.display.flip()
                
                # Control loop rate (60 FPS for pygame, 10 Hz for publishing)
                self.clock.tick(60)
                
                # Spin ROS2 once
                rclpy.spin_once(self, timeout_sec=0.01)
                
        except KeyboardInterrupt:
            self.get_logger().info("Keyboard interrupt received")
        finally:
            pygame.quit()
    
    def destroy_node(self):
        """Clean up resources"""
        pygame.quit()
        super().destroy_node()


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    try:
        node = KeyboardSignalNode()
        node.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
