# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st
from real_world.spacemouse_shared_memory import Spacemouse
from real_world.tm_robot_controller import TMRobotController
from common.precise_sleep import precise_wait


# %%
@click.command()
@click.option('-rh', '--robot_hostname', default='192.168.0.103')
@click.option('-f', '--frequency', type=float, default=10)
def main(robot_hostname, frequency):
    max_pos_speed = 250  #0.25
    max_rot_speed = 0.6
    cube_diag = np.linalg.norm([1,1,1])
    tcp_offset = 0.13
    # tcp_offset = 0
    dt = 1/frequency
    command_latency = dt / 2

    with SharedMemoryManager() as shm_manager:
        with TMRobotController(
            shm_manager=shm_manager,
            robot_ip=robot_hostname
        ) as controller,\
        Spacemouse(
            shm_manager=shm_manager
        ) as sm:
            print('Ready!')
            # to account for recever interfance latency, use target pose
            # to init buffer.
            state = controller.get_state()
            target_pose = state['TargetTCPPose']
            t_start = time.monotonic()

            iter_idx = 0
            while True:
                s = time.time()
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                precise_wait(t_sample)
                sm_state = sm.get_motion_state_transformed()
                # print(sm_state)
                dpos = sm_state[:3] * (max_pos_speed / frequency)
                drot_xyz = sm_state[3:] * (max_rot_speed / frequency)

                drot = st.Rotation.from_euler('xyz', drot_xyz)
                #print(target_pose[3], target_pose[4], target_pose[5])
                #print(drot.as_rotvec(degrees=True))
                target_pose[:3] += dpos
                target_pose[3:] = (drot * st.Rotation.from_rotvec(
                    target_pose[3:], degrees=True)).as_rotvec(degrees=True)

                #target_pose[3:] = drot.as_rotvec(degrees=True) + target_pose[3:]
                #print(target_pose[3], target_pose[4], target_pose[5])

                dpos = 0
                controller.schedule_waypoint(target_pose, 
                    t_command_target-time.monotonic()+time.time())

                precise_wait(t_cycle_end)
                iter_idx += 1
                #print(1/(time.time() -s))


# %%
if __name__ == '__main__':
    main()
