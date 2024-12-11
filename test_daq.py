

def test_basic_read():

    import nidaqmx

    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai1", min_val=-10.0, max_val=10.0)
        while True:
            r = task.read()
            print(r)


def test_read_many():

    import nidaqmx
    from nidaqmx.constants import AcquisitionType 
    from nidaqmx.stream_readers import AnalogMultiChannelReader
    import numpy as np

    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai1", min_val=-10.0, max_val=10.0)
        task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai2", min_val=-10.0, max_val=10.0)
        task.timing.cfg_samp_clk_timing(100, source="", sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=100)
        reader = AnalogMultiChannelReader(task.in_stream)

        while True:
            out = np.zeros((2,100), dtype=np.float64)
            reader.read_many_sample(data=out, number_of_samples_per_channel=100)  # read from DAQ
            np.transpose(out, axes=[1, 0])  # (2,100) -> (100,2)
            print(out)

if __name__=='__main__':
    test_read_many()
