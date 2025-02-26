

def test_gripper():
    pass
    #from real_world. import 



def test_space_mouse():

    from multiprocessing.managers import SharedMemoryManager
    from real_world.spacemouse_shared_memory import Spacemouse

    # robot
    # x+ foward, y+ left, z+ up
    # Rx+ right hand (thumb along y- axis), Ry+ right hand (thumb along x- axis), Rz+ right hand (thumb along z+ axis)

    with SharedMemoryManager() as shm_manager:
        with Spacemouse(
            shm_manager=shm_manager
        ) as sm:
            while True:
                # x+: backward, y+: right, z+: up
                # Rx+: right hand (thumb along x+), Ry+: right hand (thumb along y+), Rz+: right hand (thumb along z+)
                sm_state = sm.get_motion_state_transformed()
                print('Pos: {}, Rot: {}'.format(sm_state[:3], sm_state[3:]))

if __name__=='__main__':
    test_space_mouse()
