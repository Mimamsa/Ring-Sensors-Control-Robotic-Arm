"""
Usage:
    python3 main.py -c example/eval_dummy_config.yaml
"""
import os
import click
import yaml
import time
from multiprocessing.managers import SharedMemoryManager
from real_world.keystroke_counter import KeystrokeCounter, Key, KeyCode
from real_world.operation_env import OperationEnv


@click.command()
@click.option('--config_file', '-c', required=True, help='Path to config yaml file')
@click.option('--model_path', '-m', required=False, help='Path to Tensorflow or Pytorch model')
@click.option('--output', '-o', required=False, help='Directory to save recording')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
def main(config_file, model_path, output, frequency):

    # load robot config file
    config_data = yaml.safe_load(open(os.path.expanduser(config_file), 'r'))

    daqs_config = config_data['daqs']
    robots_config = config_data['robots']
    grippers_config = config_data['grippers']

    # setup experiment
    dt = 1/frequency

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            OperationEnv(
                daqs_config=daqs_config,
                robots_config=robots_config,
                grippers_config=grippers_config,
                output_dir=output,
                shm_manager=shm_manager) as env:

            # creating model

            # warm up model


            time.sleep(3)

            print('Ready!')
            while True:
                try:
                    # get observation
                    obs = env.get_obs()  # obs['daq_values'].shape: (100, 6)

                    # preprocess observation (maybe)
                    # obs['daq_values'] = normalize(obs['daq_values'])

                    # run inference


                    # convert model outputs to env actions


                    # execute actions
                    # env.exec_actions(
                        # actions, 
                        # timestamps,
                        # compensate_latency)

                    # visualize actions


                    # handle key presses
                    press_events = key_counter.get_press_events()
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            # Exit program
                            env.end_episode()
                            exit(0)

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    break

        print("Stopped.")


if __name__=='__main__':
    main()