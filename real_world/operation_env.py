"""The example operation environment, which includes:
- DummyDataAquisition x1
- DummyRobotController x1
- DummyGripperController x1
"""
import math
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from real_world.dummy_daq import DummyDataAquisition
from real_world.dummy_robot_controller import DummyRobotController
from real_world.dummy_gripper_controller import DummyGripperController
from common.interpolation_util import get_interp1d


class OperationEnv:

    def __init__(self,
            # required params
            daqs_config,  # list of dict[{daq_type: 'dummy', frequency: 100}]
            robots_config,  # list of dict[{robot_type: 'dummy', robot_ip: XXX, frequency: 125, obs_latency: 0.0001, action_latency: 0.1}]
            grippers_config,  # list of dict[{gripper_type: 'dummy', gripper_ip: XXX, gripper_port: 1000, frequency: 10, obs_latency: 0.01, action_latency: 0.1}]
            output_dir,
            # env params
            frequency=100,  # frequency for sampling all observations (daq only for now)
            daq_obs_horizon=100,  # input of model in time axis
            # vis params
            # enable_cam_vis=True,
            # cam_vis_resolution=(960, 960),
            # shared memory
            shm_manager=None
        ):

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        daqs: List[DummyDataAquisition] = list()
        robots: List[DummyRobotController] = list()
        grippers: List[DummyGripperController] = list()

        # Load DAQ
        for dc in daqs_config:
            if dc['daq_type'].startswith('cDAQ-V1102'):
                raise NotImplementedError()
            elif dc['daq_type'].startswith('dummy'):
                this_daq = DummyDataAquisition(
                    shm_manager = shm_manager,
                    frequency = dc['frequency'],
                    verbose = False
                )
            else:
                raise NotImplementedError()
            daqs.append(this_daq)

        # Load robotic arm
        for rc in robots_config:
            if rc['robot_type'].startswith('tm5-900'):
                raise NotImplementedError()
            elif rc['robot_type'].startswith('dummy'):
                this_robot = DummyRobotController(
                    shm_manager = shm_manager,
                    robot_ip = rc['robot_ip'],
                    frequency = rc['frequency'],
                    receive_latency=rc['robot_obs_latency'],
                    verbose = True
                )
            else:
                raise NotImplementedError()
            robots.append(this_robot)

        # Load gripper
        for gc in grippers_config:
            if gc['gripper_type'].startswith('hand-e'):
                raise NotImplementedError()
            elif gc['gripper_type'].startswith('dummy'):
                this_gripper = DummyGripperController(
                    shm_manager = shm_manager,
                    hostname = gc['gripper_ip'],
                    port = gc['gripper_port'],
                    frequency = gc['frequency'],
                    receive_latency=gc['gripper_obs_latency'],
                    use_meters = False,
                    verbose = True
                )
            else:
                raise NotImplementedError()
            grippers.append(this_gripper)

        self.daqs = daqs
        self.daqs_config = daqs_config
        self.robots = robots
        self.robots_config = robots_config
        self.grippers = grippers
        self.grippers_config = grippers_config

        self.frequency = frequency
        self.daq_obs_horizon = daq_obs_horizon
        # temp memory buffers
        self.last_daq_data = None

    # ======== start-stop API =============
    @property
    def is_ready(self):
        ready_flag = True
        for daq in self.daqs:
            ready_flag = ready_flag and daq.is_ready
        for robot in self.robots:
            ready_flag = ready_flag and robot.is_ready
        for gripper in self.grippers:
            ready_flag = ready_flag and gripper.is_ready
        return ready_flag
    
    def start(self, wait=False):
        for daq in self.daqs:
            daq.start(wait=False)
        for robot in self.robots:
            robot.start(wait=False)
        for gripper in self.grippers:
            gripper.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        # self.end_episode()
        for daq in self.daqs:
            daq.stop(wait=False)
        for robot in self.robots:
            robot.stop(wait=False)
        for gripper in self.grippers:
            gripper.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        for daq in self.daqs:
            daq.start_wait()
        for robot in self.robots:
            robot.start_wait()
        for gripper in self.grippers:
            gripper.start_wait()

    def stop_wait(self):
        for daq in self.daqs:
            daq.stop_wait()
        for robot in self.robots:
            robot.stop_wait()
        for gripper in self.grippers:
            gripper.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_obs(self, align=True) -> dict:

        obs_data = dict()

        # get daq obs
        k = math.ceil(self.daq_obs_horizon * (100 / self.frequency)) \
            + 2  # here 2 is adjustable, typically 1 should be enough
        self.last_daq_data = self.daqs[0].get_k(
            k=k, 
            out=self.last_daq_data)

        if align:
            # get last timestamp
            last_timestamp = self.last_daq_data['timestamp'][-1]
            dt = 1 / self.frequency

            # align daq obs (by linear interpolator)
            daq_obs_timestamps = last_timestamp - (
                np.arange(self.daq_obs_horizon)[::-1] * dt)  # [::-1] is reversing 1D array
            daq_obs = dict()
            daq_interpolator = get_interp1d(
                t=self.last_daq_data['timestamp'], 
                x=self.last_daq_data['daq_values'])
            daq_values = daq_interpolator(daq_obs_timestamps)
            daq_obs = {
                'daq_values': daq_values,
            }

            # update obs_data
            obs_data.update(daq_obs)

        else:
            obs_data['timestamp'] = self.last_daq_data['timestamp'][-100:]
            obs_data['daq_values'] = self.last_daq_data['daq_values'][-100:]

        return obs_data

    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray,
            compensate_latency=False):

        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)

        # execute only new actions
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]

        assert new_actions.shape[1] // len(self.robots) == 7  # 6 DoF robot + 1 gripper
        assert new_actions.shape[1] % len(self.robots) == 0

        # schedule waypoints
        for i in range(len(new_actions)):
            for robot_idx, (robot, gripper, rc, gc) in enumerate(zip(self.robots, self.grippers, self.robots_config, self.grippers_config)):
                r_latency = rc['robot_action_latency'] if compensate_latency else 0.0
                g_latency = gc['gripper_action_latency'] if compensate_latency else 0.0
                r_actions = new_actions[i, 7 * robot_idx + 0: 7 * robot_idx + 6]
                g_actions = new_actions[i, 7 * robot_idx + 6]
                robot.schedule_waypoint(
                    pose=r_actions,
                    target_time=new_timestamps[i] - r_latency
                )
                gripper.schedule_waypoint(
                    pos=g_actions,
                    target_time=new_timestamps[i] - g_latency
                )

        # record actions
        # if self.action_accumulator is not None:
            # self.action_accumulator.put(
                # new_actions,
                # new_timestamps
            # )

    def get_robot_state(self):
        return [robot.get_state() for robot in self.robots]

    def get_gripper_state(self):
        return [gripper.get_state() for gripper in self.grippers]


    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        pass


    def end_episode(self):
        "Stop recording"
        pass


    def drop_episode(self):
        pass

