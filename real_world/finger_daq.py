import time
import numpy as np
import multiprocessing as mp
import nidaqmx
from nidaqmx.constants import AcquisitionType 
from nidaqmx.stream_readers import AnalogMultiChannelReader
from multiprocessing.managers import SharedMemoryManager
from shared_memory.shared_memory_ring_buffer import (SharedMemoryRingBuffer, Empty)
from common.precise_sleep import precise_wait


class FingerDataAquisition(mp.Process):
    
    def __init__(self,
            shm_manager: SharedMemoryManager,
            frequency=100,
            get_max_k=None,
            launch_timeout=3,
            verbose=False
        ):
        super().__init__(name="FingerDataAquisition")
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.verbose = verbose
        self.keep_running = True

        # build ring buffer
        example = {
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
            print(f"[FingerDataAquisition] DAQ process spawned at {self.pid}")

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
            with nidaqmx.Task() as task:
                task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai1", min_val=-10.0, max_val=10.0)
                task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai2", min_val=-10.0, max_val=10.0)
                task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai3", min_val=-10.0, max_val=10.0)
                task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai4", min_val=-10.0, max_val=10.0)
                task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai5", min_val=-10.0, max_val=10.0)
                task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai6", min_val=-10.0, max_val=10.0)
                # Set DAQ sampling rate
                task.timing.cfg_samp_clk_timing(100, source="", sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=100)

                # Set DAQ multi-channel reader
                reader = AnalogMultiChannelReader(task.in_stream)

                if self.verbose:
                    print(f"[FingerDataAquisition] DAQ started.")

                iter_idx = 0
                t_start = time.monotonic()
                while self.keep_running:
                    t_now = time.monotonic()
                    # Generate signals
                    #daq_values = task.read()  # list of 6 float
                    #daq_values = np.array(daq_values)
                    daq_values = np.zeros((6,100), dtype=np.float64)
                    reader.read_many_sample(data=daq_values, number_of_samples_per_channel=100)  # read from DAQ

                    # stamp timestamps: latest 100 timestamps before `time.monotonic() - t_start`
                    # [::-1] is reversing 1D array
                    dt = 1 / self.frequency
                    timestamps = time.monotonic() - t_start - (np.arange(100)[::-1] * dt)

                    # Recursively save timestamps & DAQ values 
                    for i in range(100):
                        message = {
                            'timestamp': timestamps[i],
                            'daq_values': daq_values[:,i]
                        }
                        self.ring_buffer.put(message)

                    # first loop successful, ready to receive command
                    if iter_idx == 0:
                        self.ready_event.set()
                    iter_idx += 1

                    # regulate frequency
                    #dt = 1 / self.frequency
                    t_end = t_start + dt * iter_idx
                    precise_wait(t_end=t_end, time_func=time.monotonic)

                    if self.verbose:
                        print(f"[FingerDataAquisition] Actual frequency {1/(time.monotonic() - t_now)}")

        finally:
            self.ready_event.set()
            
            if self.verbose:
                print(f"[FingerDataAquisition] DAQ terminated.")
