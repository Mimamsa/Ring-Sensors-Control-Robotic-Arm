"""
"""
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from common.precise_sleep import precise_wait, precise_sleep


class TENGVisualizer(mp.Process):
    def __init__(self,
        daqs,
        daq_obs_horizon=100,
        window_name='TENG Vis',
        vis_fps=20,
        moving_window=True,
        verbose=False):
        super().__init__()

        self.fig = plt.figure()
        
        self.ax = plt.gca()
        # self.axax.set_xlim([xmin, xmax])
        self.ax.set_ylim([-1, 1])

        self.daqs = daqs
        self.daq_obs_horizon = daq_obs_horizon
        self.window_name = window_name
        self.vis_fps = vis_fps
        self.moving_window = moving_window
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

        # wati daq data
        time.sleep(3)

        last_timestamp = 0.
        while not self.stop_event.is_set():
            # Get DAQ data
            start_t = time.monotonic()
            daq_dict = self.daqs[0].get_k(k=self.daq_obs_horizon)
            timestamp = daq_dict['timestamp']
            vis_data = daq_dict['daq_values']

            # Plot only new points
            is_new = timestamp > last_timestamp
            new_timestamp = timestamp[is_new]
            new_vis_data = vis_data[is_new]
            last_timestamp = timestamp[-1]

            # Plot 6 axis data
            if self.moving_window:
                self.ax.set_xlim([last_timestamp-3, last_timestamp+1])  # keep track on latest 3 seconds
            plt.plot(new_timestamp, new_vis_data[:,0], 'bo')
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
            cv2.waitKey(1)

            # Regulate frequency
            display_latency = time.monotonic() - start_t
            if self.verbose:
                print('[TENGVisualizer] Display latency: {}'.format(display_latency))
            precise_sleep((1/self.vis_fps)-display_latency)
