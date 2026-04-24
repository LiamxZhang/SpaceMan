#
import sys
import torch
import genesis as gs

# Extension APIs
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))
from envs.genesis_env import GenesisSim

def is_finite_tensor(x) -> bool:
    if isinstance(x, (list, tuple)):
        try:
            x = torch.tensor(x, dtype=torch.float32)
        except Exception:
            return False
    if not isinstance(x, torch.Tensor):
        return False
    return torch.isfinite(x).all().item()

def check_link_pose(entity, link_name, device="cuda:0"):
    try:
        link = entity.get_link(link_name)
    except Exception as e:
        print(f"[DEBUG] Link '{link_name}' not found on entity: {e}")
        return False
    pos = torch.tensor(link.get_pos(), dtype=torch.float32, device=device)
    quat = torch.tensor(link.get_quat(), dtype=torch.float32, device=device)
    ok = torch.isfinite(pos).all().item() and torch.isfinite(quat).all().item()
    if not ok:
        print(f"[DEBUG][NaN] link={link_name} pos={pos.cpu().tolist()} quat={quat.cpu().tolist()}")
    return ok

def list_links(entity):
    try:
        names = [l.name for l in entity.links]
        print(f"[DEBUG] Links: {names}")
        return names
    except Exception as e:
        print(f"[DEBUG] Could not list links: {e}")
        return []

def main():
    GS = GenesisSim()
    scene = GS.scene
    # create franka keyboard operator
    from controllers.keyboard_controller import KeyboardController
    controller = KeyboardController()

    # from sensors.wrist_camera import WristCamera
    # wrist_camera = WristCamera()

    from robots.satellite_manipulator import SatelliteManipulator
    # franka_merge = SatelliteManipulator(name="franka_merge",sensors=[], backends=[])

    from robots.manipulator import Manipulator
    # starlink = Manipulator(name="franka",sensors=[], backends=[])

    if False:
        # runtime patch: give small positive inertials to massless links
        try:
            from utils.patch_urdf import fix_urdf_inertials
            urdf_path = (current_file_path.parent / "assets" / "urdf" / "satellite" / "urdf" / "satellite.urdf").resolve()
            fix_urdf_inertials(urdf_path, min_mass=1e-2, min_inertia=1e-4, links_to_fix=["left_Link", "attachment"])
            print(f"[DEBUG] URDF inertials patched at {urdf_path}")
        except Exception as e:
            print(f"[DEBUG] URDF inertial patch skipped: {e}")
    from pathlib import Path
    import genesis as gs

    current_file_path = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_file_path))

    to_posix = lambda p: p.as_posix()

    _ROOT_PATH = current_file_path.parent.parent
    _ASSET_PATH = _ROOT_PATH / 'src' / 'assets'

    _STARLINK_PATHS = {
        'urdf': _ASSET_PATH / 'urdf' / 'starlink' / 'urdf' / 'starlink.urdf',
    }
    from robots.robot import Robot
    starlink = Robot(name="starlink")
    # satellite_part = Robot(name="satellite_part")

    # franka = scene.add_entity(
    #     # gs.morphs.URDF(
    #     #     file='urdf/panda_bullet/panda.urdf',
    #     #     fixed=True,
    #     # ),
    #     gs.morphs.URDF(file=to_posix(_STARLINK_PATHS['urdf'])),
    # )
    # 
    GS.start()
    starlink.initialize()
    # franka.initialize()
    # satellite.initialize()
    # satellite.show_info()

    # force_tensor = torch.tensor([[0.0, 0.0, 1000.0]], dtype=torch.float32, device="cuda:0")
    # link_idx = 0

    while True:
        # franka_merge.apply_force(force=control_pos, torque=control_orien, link_name='starlink_base_star_link')
        starlink.apply_force(force=[0.0, 0.0, 1000.0], link_name='base')
        # franka_merge.apply_force(force=[0.0, 0.0, 10000.0], torque=[1000.0, 0.0, 0.0], link_name='starlink_base_star_link')
        # franka_merge.apply_force(torque=[0.0, 1000.0, 0.0], link_name='starlink_base_star_link')
        # franka_merge.apply_force(torque=[0.0, 0.0, 1000.0], link_name='starlink_base_star_link')

        # scene.sim.rigid_solver.apply_links_external_force(force=force_tensor, links_idx=link_idx)
        # scene.step()

        # print(control_pos, control_orien)
        # print(franka_merge.ee_state.link_parent_global_state)
        starlink.step()
        # franka.step()
        # satellite.apply_force(force=[100000000, 0, 0], link_name='base_link')
        # satellite.step()
        GS.step()
        if not GS.viewer.is_alive(): #
            print("Viewer window has been closed.")
            break

    GS.stop()

if __name__ == "__main__":
    main()
