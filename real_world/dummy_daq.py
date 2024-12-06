import time
import numpy as np
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from shared_memory.shared_memory_ring_buffer import (SharedMemoryRingBuffer, Empty)
from common.precise_sleep import precise_wait


class DummyDataAquisition(mp.Process):
    
    def __init__(self,
            shm_manager: SharedMemoryManager,
            frequency=100,
            get_max_k=None,
            launch_timeout=3,
            verbose=False
        ):
        super().__init__(name="DummyDataAquisition")
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.verbose = verbose
        self.keep_running = True

        # build ring buffer
        example = {
            'id': 0,
            'timestamp': 0.,
            # from 0 to 5 order: 'lthumb', 'rthumb', 'lindex', 'rindex', 'lmid', 'rmid'
            'daq_values': np.zeros((6,), dtype=np.float64),
        }

        if get_max_k is None:
            get_max_k = int(frequency * 10)

        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[DummyDataAquisition] DAQ process spawned at {self.pid}")

    def stop(self, wait=True):
        self.keep_running = False
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

    # ========= receive APIs =============
    def get(self, out=None):
        return self.ring_buffer.get(out=out)

    def get_k(self, k=None, out=None):
        return self.ring_buffer.get_last_k(k=k, out=out)

    # ========= main loop in process ============
    def run(self):

        try:
            # -- CONNECT TO A NI DAQ THIS LINE --
            if self.verbose:
                print(f"[DummyDataAquisition] DAQ started.")

            iter_idx = 0
            t_start = time.monotonic()
            while self.keep_running:
                t_now = time.monotonic()
                # Generate signals
                # -- GET DATA FROM NI DAQ HERE --
                message = {
                    'id': iter_idx,
                    'timestamp': time.monotonic() - t_start,
                    # from 0 to 5 order: 'lthumb', 'rthumb', 'lindex', 'rindex', 'lmid', 'rmid'
                    'daq_values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
                }
                self.ring_buffer.put(message)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                # regulate frequency
                dt = 1 / self.frequency
                t_end = t_start + dt * iter_idx
                precise_wait(t_end=t_end, time_func=time.monotonic)

                if self.verbose:
                    print(f"[DummyDataAquisition] Actual frequency {1/(time.monotonic() - t_now)}")

        finally:
            self.ready_event.set()
            
            if self.verbose:
                print(f"[DummyDataAquisition] DAQ terminated.")