
"""
Configuration for starlink_combine_qf_space_manipulator URDF model.
"""
from pathlib import Path
import genesis as gs
import torch
to_posix = lambda p: p.as_posix()

_current_file_path = Path(__file__).resolve().parent
_ROOT_PATH = _current_file_path.parent.parent
_ASSET_PATH = _ROOT_PATH / 'src' / 'assets'

_STARLINK_MANIPULATOR_URDF = (
    _ASSET_PATH / 'urdf' / 'starlink_combine_qf_space_manipulator' 
    / 'starlink_combine_qf_space_manipulator.urdf'
)

FRANKA_S_Q_CONFIG = {
    "initial_dofs": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04, 0.04, 0.04, 0.04],
    "control": {
        "kp": [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100, 100, 100, 100],
        "kv": [450, 450, 350, 350, 200, 200, 200, 10, 10, 10, 10, 10],
        "force_range_min": [-87, -87, -87, -87, -12, -12, -12, -100, -100, -100, -100, -100],
        "force_range_max": [87, 87, 87, 87, 12, 12, 12, 100, 100, 100, 100, 100],
    }
}

IK_PARAMS = {
    "smooth_factor": 0.3,
    "max_joint_change": 0.05,
}

FRANKA_S_Q_PARAMS = {
    "name": "franka_merge",
    "end_effector": "qf_space_manipulator_2F-Body_Link",
    "base": "starlink_base_star_link",
    "config": FRANKA_S_Q_CONFIG,
    "joints": (
        # "root_joint",
        "qf_space_manipulator_Shoulder_link_1_yaw",
        "qf_space_manipulator_Upper_arm_Link_1_roll",
        "qf_space_manipulator_Mid_arm_Link_1_roll",
        "qf_space_manipulator_Upper_wrist_Link_1_yaw",
        "qf_space_manipulator_Upper_wrist_Link_2_roll",
        "qf_space_manipulator_2F-Body_Link_pitch", 
        # hand
        "qf_space_manipulator_Finger1_2_Link_roll",
        "qf_space_manipulator_Finger3_2_Link_roll", # all gripper joints mimic this
        "qf_space_manipulator_Finger3_1_Link_roll",
        "qf_space_manipulator_Finger1_1_Link_roll", # the only positive mutiplier
        "qf_space_manipulator_Finger4_2_Link_roll",
        "qf_space_manipulator_Finger4_1_Link_roll",
    ),
    "motor": 6,
    "finger": 12,
    "gripper_revolute": [-1, 1, -1, 1, -1, -1],
    # "finger_open": [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2],
    # "finger_close": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
    "finger_open": [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
    "finger_close": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    "ik_params": IK_PARAMS,
    "path": _STARLINK_MANIPULATOR_URDF,
}

FRANKA_S_Q_PID = {
    "name": "franka_merge",
    "enable_pid": True,  # 控制是否启用PID控制器
    "P": [1000, 1000, 1000, 1000, 0, 0],
    "I": [0, 0, 0, 0, 0, 0],
    "D": [1000, 1000, 1000, 100, 0, 0],
    "setpoint": [0, 0, 0, 0, 0, 0],
    "dt": 0.01,
    "limits": [
        500,         # x
        None,        # y
        None,        # z
        None,        # roll
        None,        # pitch
        None         # yaw
    ]
}

FRANKA_S_Q_CAMERA = {
    "name": "franka",
    "wrist_camera": True,
    "camera": {
        "res": (640, 480),
        "pos": (-1, -1, -1),
        "lookat": (0, 0, 0),
        "fov": 70,
        "GUI": True
    },
    "enable_recording": False,  # 控制是否启用录制的flag
    "end_effector_link": "qf_space_manipulator_2F-Body_Link",
    "pos_offset": torch.tensor([0.07, 0.1, -0.1], dtype=torch.float32),
    "lookat_offset": torch.tensor([0.1, -1.0, 0.0], dtype=torch.float32),
    "up_offset": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
}

def _make_starlink_manipulator():
    return {
        "morph": gs.morphs.URDF(
            file=to_posix(_STARLINK_MANIPULATOR_URDF),
            pos=(-0.3, 0.0, 0.0),
            euler=(0.0, 0.0, 0.0),
            merge_fixed_links=False,
            fixed=False,
        ),
        "material": gs.materials.Rigid(
            gravity_compensation=1.0,
        ),
    }

__all__ = [
    'FRANKA_S_Q_PARAMS',
    'FRANKA_S_Q_PID',
    'FRANKA_S_Q_CAMERA',
    '_make_starlink_manipulator',
]