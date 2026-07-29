# X2 VR demonstration recorder

This directory contains the data path that sits beside, not inside, the X2
real-time controller:

```text
PICO/XRoboToolkit -> GMR teleop bridge -> rl_tracking
                         | recorder PUB tap
                         v
                 x2_vr_recorder.py
                         ^
                         | ROS 2 camera/joints/IMU
                         |
                       X2 robot
```

Code belongs in `motion_tracking/sim2real/data_collection`.  Large recordings
default to `~/Datasets/x2_vr/` and must not be committed to Git.

## Why the legacy recorder must not run on the robot path

Do **not** run `teleop/record_teleop_retarget_zmq.py` while C++ VR control is
active.  Ports 28702 and 28703 use ZeroMQ PUSH/PULL.  A second PULL consumer
load-balances rather than copies messages, so the old recorder can steal
reference and button messages from `VRMotionSource`.

The updated bridge publishes a separate, non-blocking PUB tap on port 28704.
The controller still has exclusive use of 28701-28703.

## Recorded streams

Each event retains both its device/source timestamp and the recorder/bridge
wall and monotonic receive timestamps.

- `xr`: 24 raw XR body poses, headset pose and controller state.
- `retarget`: the latest raw GMR `root xyz + quaternion wxyz + q29` result.
- `reference`: the interpolated frame actually sent by the Python bridge to
  C++ (before C++ yaw/position anchoring and transition blending).
- `controller`: 50 Hz normalized PICO button state.
- `camera_head`: original compressed X2 head-camera frame.
- `joint_states`: split leg/waist/arm/head `sensor_msgs/JointState` messages.
- `imu_torso`, `imu_chest`: original `sensor_msgs/Imu` data.

The 29-joint action order is copied from `rl_tracking.yaml/BaseConfig.seq`:
legs, waist `yaw/pitch/roll`, then each arm with wrist `yaw/pitch/roll`.

## 1. Restart the VR bridge with the recorder tap

The tap is disabled during normal teleoperation so it adds no serialization
load to the latency-sensitive path.  For a recording session, add:

```bash
--tap_bind_addr tcp://*:28704
```

to the normal X2 bridge command.  After restarting, it should print:

```text
[teleop_tap] publishing recorder events at tcp://*:28704
```

The bridge queues tap events without blocking control.  Every tap event also
has a continuous `tap_seq`; the recorder reports transport gaps.

## 2. Check that the laptop can see robot ROS topics

The robot uses ROS domain 0 by default.  On the laptop:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 topic hz /aima/hal/sensor/rgbd_head_front/rgb_image/compressed
ros2 topic echo /joint_states/leg --once
ros2 topic echo /imu/torso/data --once
```

Do not start a real recording until all three are visible.  If only the camera
is missing, check the camera service and the robot/laptop DDS network profile.

## 3. Run the raw recorder on the laptop

Source ROS before invoking the existing GMR virtualenv so `rclpy` and
`sensor_msgs` are visible:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

cd ~/Documents/motion_tracking/sim2real/data_collection
~/Documents/gmr/.venv/bin/python x2_vr_recorder.py \
  --task "touch the red button with the left hand" \
  --output_root ~/Datasets/x2_vr/raw
```

The camera subscription defaults to ROS sensor-style `best_effort` QoS.  If
`ros2 topic info -v` shows that your camera publisher is reliable, add
`--camera_qos_reliability reliable`.

Before the first right-button press, wait for non-zero `controller`,
`camera_head`, `joint_states`, and `imu_torso` counts.  `reference` normally
starts growing only after the right button enables VR and C++ begins requesting
frames; do not wait for it beforehand.

- Right `key_one`: start a new episode.
- Left `key_one`: stop, capture the configured post-roll, flush, and save.
- `Ctrl+C`: preserve an active episode with status `interrupted`.

If any required stream was absent, raw data is still preserved but the
manifest is marked `status: invalid`, so it cannot silently enter conversion.
Validation also checks all 29 leg/waist/arm joints, required split joint topics,
minimum active-window frame counts, and long stream gaps.

Make the first trial only 5--10 seconds.  While it runs, confirm `reference`
starts increasing and `queue` remains far below its limit with no drops; after
the left-button stop, immediately run the synchronization dry-run below.

Reference-only diagnostics are possible with `--disable_ros`, but those
episodes do not contain enough observation data for policy training.

## Raw layout

During recording, the directory is hidden and marked partial.  On a clean stop
it is atomically renamed:

```text
~/Datasets/x2_vr/raw/
└── episode_000000/
    ├── manifest.json
    ├── streams/
    │   ├── camera_head.jsonl
    │   ├── controller.jsonl
    │   ├── imu_torso.jsonl
    │   ├── joint_states.jsonl
    │   ├── reference.jsonl
    │   ├── retarget.jsonl
    │   └── xr.jsonl
    └── images/head_rgb/*.jpg
```

The recorder does not assume that pressing stop means success.  Annotate an
episode after reviewing it:

```bash
~/Documents/gmr/.venv/bin/python annotate_episode.py \
  ~/Datasets/x2_vr/raw/episode_000000 --success true
```

## 4. Validate synchronization

This does not require LeRobot:

```bash
~/Documents/gmr/.venv/bin/python convert_to_lerobot.py \
  --raw_root ~/Datasets/x2_vr/raw \
  --fps 25 \
  --dry_run
```

Inspect the accepted-frame ratio and the missing/stale counters before
collecting many demonstrations.

## 5. Convert to LeRobotDataset v3

The realtime GMR environment is Python 3.10 and should remain small and
stable.  The pinned LeRobot v0.6 requires Python 3.12, so use a separate converter
environment:

```bash
uv python install 3.12
uv venv --python 3.12 ~/venvs/lerobot-v3
source ~/venvs/lerobot-v3/bin/activate
uv pip install 'lerobot[dataset]==0.6.0'

cd ~/Documents/motion_tracking/sim2real/data_collection
python convert_to_lerobot.py \
  --raw_root ~/Datasets/x2_vr/raw \
  --output_root ~/Datasets/x2_vr/lerobot_v3 \
  --repo_id local/x2_vr \
  --fps 25
```

The X2 head camera is mounted upside down for this stream, so conversion rotates
frames by 180 degrees by default. Raw compressed images remain unchanged. Pass
`--camera_rotation_deg 0` if the camera publisher has already corrected them.

Use `--require_success` after the episodes have been annotated.  The converter
produces:

- `observation.images.head`: RGB video;
- `observation.state`: q29, dq29 and torso IMU (68 floats);
- `action`: bridge reference `root xyz + quaternion wxyz + q29` (36 floats);
- one language task string per episode.

When a synchronized tick is missing, the converter splits at that gap instead
of concatenating the remaining frames and speeding up time.  Existing output
directories are never deleted or overwritten.

Here `action` means the high-level whole-body reference that a future VLA can
generate and then feed through the existing C++ alignment plus RL tracking
policy.  It is deliberately **not** the final actuator command: the raw
reference is captured before C++ root/yaw anchoring and start blending.  A
model intended to bypass the tracking controller would additionally need the
aligned/consumed reference or `/joint_cmd/*` streams.

The raw data remains the source of truth, so alternative local/delta action
representations can be generated later without repeating a demonstration.

Run the bridge and recorder on the same laptop.  Synchronization uses the
recorder's monotonic receive clock; device, ROS, and bridge timestamps remain
in the raw files for latency diagnostics.
