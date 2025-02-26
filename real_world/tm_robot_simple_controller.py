import time
import enum
import numpy as np
import multiprocessing as mp
import socket
import math
from multiprocessing.managers import SharedMemoryManager
from shared_memory.shared_memory_queue import (SharedMemoryQueue, Empty)
from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from common.precise_sleep import precise_wait
from common.tm_script_wrapper import move_coordinate, get_tool_coord


class Command(enum.Enum):
    STOP = 0
    SCHEDULE_WAYPOINT = 1


class TMRobotController(mp.Process):
    def __init__(self,
            shm_manager: SharedMemoryManager,
            robot_ip, 
            frequency=10,
            max_pos_speed=0.1,
            max_rot_speed=0.1,
            get_max_k=None,
            command_queue_size=256,
            launch_timeout=3,
            receive_latency=0.0,
            verbose=False
        ):
        super().__init__(name="TMRobotController")
        self.tmsct_port = 5890
        self.tmsvr_port = 5891
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.verbose = verbose

        # build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'target_time': 0.
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )

        # build ring buffer
        example = {
            'ActualTCPPose': np.zeros(6,),
            # 'ActualTCPSpeed': np.zeros(6,),
            # 'ActualQ': np.zeros(6,),
            # 'ActualQd': np.zeros(6,),
            'TargetTCPPose': np.zeros(6,),
            # 'TargetTCPSpeed': np.zeros(6,),
            # 'TargetQ': np.zeros(6,),
            # 'TargetQd': np.zeros(6,)
        }
        example['robot_receive_timestamp'] = time.time()
        example['robot_timestamp'] = time.time()

        if get_max_k is None:
            get_max_k = int(frequency * 10)

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.tmsct_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tmsvr_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[TMRobotController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= main loop in process ============
    def run(self):

        try:
            # Connect to TMSCT, TMSVR
            self.tmsct_sock.connect((self.robot_ip, self.tmsct_port))
            self.tmsvr_sock.connect((self.robot_ip, self.tmsvr_port))
            if self.verbose:
                print(f"[TMRobotController] Connect to robot: {self.robot_ip}")

            # main loop
            dt = 1. / self.frequency
            curr_pose = get_tool_coord(self.tmsvr_sock)
            
            fixed_orientation = np.array(curr_pose[3:])  # degree
            curr_pose[3] *= (math.pi/180)  # deg -> rad
            curr_pose[4] *= (math.pi/180)
            curr_pose[5] *= (math.pi/180)
            #curr_pose = [0., 0., 0., 0., 0., 0.]  # get robot current TCP pose
            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            while keep_running:
                # send command to robot
                t_now = time.monotonic()
                pose_command = pose_interp(t_now)
                #print(pose_command[3], pose_command[4], pose_command[5])
                #pose_command[3:] = pose_command[3:] * (180/math.pi)  # rad -> deg
                pose_command[3:] = fixed_orientation
                move_coordinate(
                    self.tmsct_sock,
                    positions=pose_command,
                    velocity=100,
                    acc_time=200,
                    blend_percentage=0,
                    final_goal=False)

                # update robot state
                state = dict()
                state['TargetTCPPose'] = np.array(pose_command)
                state['ActualTCPPose'] = np.array(get_tool_coord(self.tmsvr_sock))
                t_recv = time.time()
                state['robot_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    # process at most 1 command per cycle to maintain frequency
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])

                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[TMRobotController] Actual frequency {1/(time.monotonic() - t_now)}")

        finally:
            # terminate
            self.tmsct_sock.close()
            self.tmsvr_sock.close()
            self.ready_event.set()

            if self.verbose:
                print(f"[TMRobotController] Disconnected from robot: {self.robot_ip}")
