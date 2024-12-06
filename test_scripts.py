"""

"""


def testSharedMemoryQueue():

    from multiprocessing.managers import SharedMemoryManager
    from shared_memory.shared_memory_queue import (SharedMemoryQueue, Empty)

    shm_manager = SharedMemoryManager()
    shm_manager.start()

    example = {
        'a': 0.0,
        'b': 0.0
    }

    input_queue = SharedMemoryQueue.create_from_examples(
        shm_manager=shm_manager,
        examples=example,
        buffer_size=10
    )

    for i in range(10):
        message = {'a': float(i)}
        input_queue.put(message)

    result = input_queue.get_k(5)
    print(result)

    shm_manager.shutdown()


def testDummyDataAquisition():

    from multiprocessing.managers import SharedMemoryManager
    from common.precise_sleep import precise_wait, precise_sleep
    from real_world.dummy_daq import DummyDataAquisition

    shm_manager = SharedMemoryManager()
    shm_manager.start()
    with DummyDataAquisition(shm_manager, frequency=100) as daq:
        while True:
            precise_sleep(0.1)
            if daq.is_ready:
                print(daq.get_k(2))

    shm_manager.shutdown()


def testDummyRobotController():

    import time
    from multiprocessing.managers import SharedMemoryManager
    from common.precise_sleep import precise_wait, precise_sleep
    from real_world.dummy_robot_controller import DummyRobotController

    shm_manager = SharedMemoryManager()
    shm_manager.start()
    with DummyRobotController(
        shm_manager = shm_manager,
        robot_ip = '0.0.0.0',
        frequency=40,
        verbose=True
    ) as robot:

        for i in range(40):
            pose = [0.05*i,0,0,0,0,0]
            target_time = time.time() + 0.2
            robot.schedule_waypoint(pose, target_time)
            precise_sleep(0.1)

        precise_sleep(0.5)
        print(robot.get_all_state())

    shm_manager.shutdown()


def testDummyGripperController():

    import time
    from multiprocessing.managers import SharedMemoryManager
    from common.precise_sleep import precise_wait, precise_sleep
    from real_world.dummy_gripper_controller import DummyGripperController

    shm_manager = SharedMemoryManager()
    shm_manager.start()
    with DummyGripperController(
        shm_manager = shm_manager,
        hostname='0.0.0.0',
        port = 0,
        frequency=10,
        use_meters=False,
        verbose=True
    ) as gripper:

        for i in range(20):
            pos = 3*i
            target_time = time.time() + 0.1
            gripper.schedule_waypoint(pos, target_time)
            precise_sleep(0.2)

        precise_sleep(1)
        print(gripper.get_all_state())

    shm_manager.shutdown()


def test_get_interp1d():

    import numpy as np
    from common.interpolation_util import get_interp1d

    t = np.array([0, 0.2, 0.4, 0.6, 0.8])
    # x = np.array([[5,5,5,5,5],
                  # [1,2,3,4,5]])
    x = np.array([[5,1],[5,2],[5,3],[5,4],[5,5]])
    print(t.shape, x.shape)

    new_t = np.array([0.3, 0.5, 0.7])
    interp = get_interp1d(t=t, x=x)
    new_x = interp(new_t)
    
    print(new_x, new_x.shape)


if __name__=='__main__':
    test_get_interp1d()