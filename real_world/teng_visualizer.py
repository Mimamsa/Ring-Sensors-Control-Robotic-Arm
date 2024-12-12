"""
"""
import time
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from common.precise_sleep import precise_wait, precise_sleep


class TENGVisualizer(mp.Process):
    def __init__(self,
        daqs,
        window_name='TENG Vis',
        vis_fps=10,
        display_window_size=200, # samples
        daq_frequency=100,
        verbose=False):
        super().__init__()

        self.fig = plt.figure()
        self.ax = plt.gca()
        self.ax.set_ylim([-0.05, 0.05])

        self.daqs = daqs
        self.window_name = window_name
        self.vis_fps = vis_fps
        self.display_window_size = display_window_size
        self.daq_frequency = daq_frequency
        self.verbose = verbose
        # shared variables
        self.stop_event = mp.Event()

    # ========= launch method ===========
    def start(self, wait=False):
        super().start()
    
    def stop(self, wait=False):
        self.stop_event.set()
        if wait:
            self.stop_wait()

    def start_wait(self):
        pass

    def stop_wait(self):
        self.join()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= main loop in process ============
    def run(self):

        cv2.setNumThreads(1)

        # Wait for DAQ data generated
        time.sleep(2)

        # Display buffer
        viz_timestamps = np.zeros((self.display_window_size,), dtype=np.float64)
        viz_daq_values = np.zeros((self.display_window_size, 6), dtype=np.float64)

        last_timestamp = 0.
        while not self.stop_event.is_set():
            # Clear axis
            # If axis is not cleared in each iteration, the display latency gains over time.
            plt.cla()
        
            # Get DAQ data
            start_t = time.monotonic()
            k = math.ceil(self.daq_frequency/self.vis_fps) + 10
            daq_dict = self.daqs[0].get_k(k=k)
            timestamps = daq_dict['timestamp']
            daq_values = daq_dict['daq_values']

            # Accumulate display data
            is_new = timestamps > last_timestamp
            new_timestamps = timestamps[is_new]
            new_daq_values = daq_values[is_new]
            last_timestamp = timestamps[-1]

            viz_timestamps = np.concatenate((viz_timestamps, new_timestamps), axis=0)
            viz_daq_values = np.concatenate((viz_daq_values, new_daq_values), axis=0)
            viz_timestamps = viz_timestamps[-self.display_window_size:]
            viz_daq_values = viz_daq_values[-self.display_window_size:,:]

            self.ax.set_ylim([-0.05, 0.05])
            self.ax.set_xlim([last_timestamp - (self.display_window_size/self.daq_frequency),
                last_timestamp+0.5])  # keep track on latest 2 seconds

            # Plot only the first channel(for testing)
            plt.plot(viz_timestamps, viz_daq_values[:,0], 'bo')
            # plot all 6 channels
            #plt.plot(new_timestamp, new_vis_data[:,0], 'bo-',
            #    new_timestamp, new_vis_data[:,1], 'go-',
            #    new_timestamp, new_vis_data[:,2], 'ro-',
            #    new_timestamp, new_vis_data[:,3], 'co-',
            #    new_timestamp, new_vis_data[:,4], 'mo-',
            #    new_timestamp, new_vis_data[:,5], 'yo-')
            self.fig.canvas.draw()

            # Converting matplotlib figure to OpenCV image
            ncols, nrows = self.fig.canvas.get_width_height()
            plot = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(nrows, ncols, 3)
            plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)

            # Displaying the image:
            cv2.imshow(self.window_name, plot)
            cv2.waitKey(1)  # wait 1 ms

            # Regulate frequency
            display_latency = time.monotonic() - start_t
            if self.verbose:
                print('[TENGVisualizer] Display latency: {}'.format(display_latency))
            sleep_time = (1/self.vis_fps) - display_latency  # seconds
            if sleep_time < 0:
                print('[TENGVisualizer] Please reduce the vis_fps.')
            else:
                precise_sleep(sleep_time)
