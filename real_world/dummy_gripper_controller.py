import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from shared_memory.shared_memory_queue import (SharedMemoryQueue, Empty)
from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from common.precise_sleep import precise_wait


class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1


class DummyGripperController(mp.Process):
    def __init__(self,
            shm_manager: SharedMemoryManager,
            hostname,
            port=63352,
            frequency=10,
            home_to_open=True,
            move_max_speed=150.0,  # mm/s
            get_max_k=None,
            command_queue_size=256,
            launch_timeout=3,
            receive_latency=0.0,
            use_meters=False,
            verbose=False
        ):
        super().__init__(name="DummyGripperController")
        self.hostname = hostname
        self.port = port
        self.frequency = frequency
        self.home_to_open = home_to_open
        self.move_max_speed = move_max_speed
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.scale = 1000.0 if use_meters else 1.0
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 10)
        
        # build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )

        # build ring buffer
        example = {
            # 'gripper_status': 0,
            # 'gripper_object_status': 0,
            # 'gripper_fault_status': 0,
            'gripper_position': 0.0,
            # 'gripper_velocity': 0.0,
            # 'gripper_force': 0.0,
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[DummyGripperController] Controller process spawned at {self.pid}")


    def stop(self, wait=True):
        message = {
            'cmd': Command.SHUTDOWN.value
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
    def schedule_waypoint(self, pos: float, target_time: float):
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
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
        # start connection
        try:
            # create intance
            if self.verbose:
                print("[DummyGripperController] Connect to gripper: {self.hostname}")
            # -- CONNECT TO A GRIPPER THIS LINE --

            # home gripper to initialize
            if self.verbose:
                print("[DummyGripperController] Activating gripper...")
            # -- ACTIVATING GRIPPER THIS LINE --
            if self.home_to_open:
                # -- HOME GRIPPER TO OPEN THIS LINE --
                pass

            # -- GET GRIPPER INITIAL POSITION THIS LINE --
            curr_pos = 60.  # mm, open distance
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[[curr_pos,0,0,0,0,0]]
            )

            keep_running = True
            t_start = time.monotonic()
            iter_idx = 0
            while keep_running:
                # command gripper
                t_now = time.monotonic()
                dt = 1 / self.frequency
                t_target = t_now
                target_pos = pose_interp(t_target)[0]
                target_vel = (target_pos - pose_interp(t_target - dt)[0]) / dt
                # -- SEND A COMMAND TO GRIPPER THIS LINE --

                # get state
                # -- UPDATE STATE DICTIONARY HERE --
                state = {
                    # 'gripper_status': gripper.get_gripper_status(),
                    # 'gripper_object_status': gripper.get_object_status(),
                    # 'gripper_fault_status': gripper.get_fault_status(),
                    # 'gripper_position': gripper.get_current_position() / self.scale,
                    'gripper_position': target_pos / self.scale,
                    # 'gripper_velocity': gripper.get_current_speed() / self.scale,
                    # 'gripper_force': gripper.get_current_force(),
                    'gripper_receive_timestamp': time.time(),
                    'gripper_timestamp': time.time() - self.receive_latency
                }
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pos = command['target_pos'] * self.scale
                        target_time = command['target_time']
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=[target_pos, 0, 0, 0, 0, 0],
                            time=target_time,
                            max_pos_speed=self.move_max_speed,
                            max_rot_speed=self.move_max_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                # regulate frequency
                dt = 1 / self.frequency
                t_end = t_start + dt * iter_idx
                precise_wait(t_end=t_end, time_func=time.monotonic)

        finally:
            # -- DISCONNECT GRIPPER THIS LINE--
            self.ready_event.set()
            if self.verbose:
                print(f"[DummyGripperController] Disconnected from gripper: {self.hostname}")




