# Ring Sensors Control Robotic Arm

This repository provide a tiny message passing framework to get data from sensors and manipulate robotic arms in real-world environment.

Features:
- Resample at different data rate for each sensors & controllers.
- Trajectory interpolation controllers.
- Get/control latency compensate.


## Installation

```
pip install -r requirements.txt
```


## Usage

```
python3 main.py -c example/eval_dummy_config.yaml
```


## References

[Universal Manipulation Interface](https://github.com/real-stanford/universal_manipulation_interface)