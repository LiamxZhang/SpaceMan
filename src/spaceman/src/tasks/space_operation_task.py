"""SpaceOperationTask: Combined simulation of a Starlink satellite bus and its onboard manipulator.

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


class SpaceOperationTask(Task):
    def __init__(self, config: Optional[dict] = None):
        super().__init__(task_name="space_operation")
        # Optional external config (not heavily used right now)
        self.user_config = config or {}
        # Hold references to spawned entities
        self.starlink: Optional[Robot] = None
        self.starlink_manipulator: Optional[SatelliteManipulator] = None

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
            self.logger.info(f"SpaceOperationTask initialized successfully")
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
    """Test SpaceOperationTask: initialize, run a few steps, then clean up."""
    print("=== SpaceOperationTask Test ===")
    
    task = SpaceOperationTask()
    
    # Initialize the task
    if not task.initialize():
        print("❌ Task initialization failed")
        return
    
    print("✅ Task initialized successfully")
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