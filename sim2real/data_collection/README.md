# X2 VR demonstration recorder

This directory contains the data path that sits beside, not inside, the X2
real-time controller:

```text
PICO/XRoboToolkit -> GMR teleop bridge -> rl_tracking
                         | recorder PUB tap
                         v
                 x2_vr_recorder.py
                    ^             ^
       camera TCP tap |             | ROS 2 joints/IMU/hand status
                    X2 vision bridge
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
- `hand_command`: authoritative mapped command status from
  `/vr_hand_controller/status`: `active` plus independent left/right grasp
  fractions in `[0, 1]`. This records the high-level command actually accepted
  by the hand controller, not all OmniHand joint targets.
- `camera_head`: original compressed X2 head-camera frame, normally received
  from the robot-side vision bridge TCP tap on port 28706.
- `joint_states`: normalized leg/waist/arm/head state.  Real X2 recording reads
  `aimdk_msgs/JointStateArray` directly; the stored JSON keeps the same flat
  name/position/velocity/effort representation used by the converter.
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

## 2. Start the camera tap and check robot sensors

Start the deployed vision bridge on the robot normally.  Its default
configuration enables the recorder tap on `0.0.0.0:28706`; look for:

```text
camera recorder tap listening on 0.0.0.0:28706
```

The tap forwards the original compressed ROS image, keeps at most the newest
pending frame, and never performs socket work in the camera callback.  Verify
the robot is listening with `ss -ltnp | grep 28706`.

The robot uses ROS domain 0 by default.  On the laptop:

```bash
source /opt/ros/humble/setup.bash
source ~/Documents/motioncontrol/x1-motion-control/x1_digit_mc/install/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 topic echo /aima/hal/joint/leg/state --once
ros2 topic echo /aima/hal/imu/torso/state --once
ros2 topic echo /vr_hand_controller/status --once
```

The AimDK camera publisher is bound to a robot-internal Fast DDS interface, so
the laptop does not need to discover its ROS topic.  The recorder receives the
compressed frame over the separate TCP tap while joints and IMUs still use ROS.
PICO `Listen` is not required: it controls only the independent H.264 headset
output, not the recorder tap.

## 3. Run the raw recorder on the laptop

Source ROS and the built motion-control workspace before invoking the existing
GMR virtualenv.  The workspace makes `aimdk_msgs/JointStateArray` type support
available on the laptop:

```bash
source /opt/ros/humble/setup.bash
source ~/Documents/motioncontrol/x1-motion-control/x1_digit_mc/install/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

cd ~/Documents/motion_tracking/sim2real/data_collection
~/Documents/gmr/.venv/bin/python x2_vr_recorder.py \
  --tap_addr tcp://172.66.88.241:28704 \
  --camera_tap_addr tcp://172.66.88.241:28706 \
  --task "touch the red button with the left hand" \
  --output_root ~/Datasets/x2_vr/raw
```

With `--camera_tap_addr` enabled, the recorder automatically skips its ROS
camera subscription but retains all ROS joint and IMU subscriptions.  The
camera client reconnects after a bridge/network interruption, and sequence
gaps are recorded as `camera_tap_transport` ingress drops.  Replace
`172.66.88.241` if the robot IP changes.

The default `--sensor_profile aimdk` reads the real robot topics directly:

```text
/aima/hal/joint/{leg,waist,arm,head}/state
/aima/hal/imu/{torso,chest}/state
```

For simulation or an older deployment that publishes the compatibility topics,
pass `--sensor_profile compat` to use `/joint_states/*` and `/imu/*/data`.

Without `--camera_tap_addr`, the legacy direct ROS camera subscription remains
available and defaults to sensor-style `best_effort` QoS.  If
`ros2 topic info -v` shows that your camera publisher is reliable, add
`--camera_qos_reliability reliable`.

Before the first right-button press, wait for non-zero `controller`,
`hand_command`, `camera_head`, `joint_states`, and `imu_torso` counts.
`reference` normally starts growing only after the right button enables VR and
C++ begins requesting frames; do not wait for it beforehand. New recordings
require `hand_command`; if the status topic is missing, the episode is retained
but marked invalid instead of silently losing the gripper action. At least two
`hand_command` events inside the A-to-X active window must also have
`active=true`; merely receiving inactive status heartbeats is not enough.

- Right `key_one`: start a new episode.
- Left `key_one`: stop, capture the configured post-roll, flush, and save.
- `Ctrl+C`: preserve an active episode with status `interrupted`.

If any required stream was absent, raw data is still preserved but the
manifest is marked `status: invalid`, so it cannot silently enter conversion.
Validation also checks all 29 leg/waist/arm joints, required split joint topics,
minimum active-window frame counts, long stream gaps, and configurable matching
event predicates. A failed predicate is listed by name in
`validation.failed_matching_event_requirements` (for hand control the name is
`hand_command_active`) and is printed when the episode is finalized.

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
    │   ├── hand_command.jsonl
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
- `action`: bridge reference `root xyz + quaternion wxyz + q29`, followed by
  left and right grasp fractions (38 floats total);
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

Hand commands use causal zero-order-hold synchronization: a dataset tick uses
only the newest status at or before that tick, never a closer future status.
For authoritative status, adjacent uint32 `sequence` values are also checked.
If events were lost, ticks strictly between the two surviving event timestamps
are rejected as `hand_command_sequence_gap` instead of silently holding an old
grasp value; each surviving event remains valid at its own timestamp (including
normal uint32 wraparound).
The default maximum age is 100 ms (`--max_hand_command_age_ms`). Inactive,
invalid, missing or stale status ticks are dropped and therefore split
continuous output segments. For old raw episodes without `hand_command`, the
converter emits an explicit warning and derives both values from the legacy
`controller` grip fields using the live mapping
`clip((grip - 0.10) / (0.90 - 0.10), 0, 1)`. This fallback represents operator
intent, not proof that the C++ hand node was armed, so authoritative new
recordings should be preferred.

Synchronization uses the recorder laptop's monotonic receive clock; device,
ROS, bridge and camera-tap timestamps remain in the raw files for latency
diagnostics.
