import time
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from shared_memory.shared_memory_ring_buffer import (SharedMemoryRingBuffer, Empty)
from common.precise_sleep import precise_wait


class DummyInference(mp.Process):

    def __init__(self,
            shm_manager: SharedMemoryManager,
            frequency=5
        ):
        super().__init__(name="DummyInference")
        
        