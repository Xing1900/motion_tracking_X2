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

## 2. Choose the camera path and check robot sensors

When the recorder itself runs on the robot, prefer the direct AimDK ROS topic:

```text
/aima/hal/sensor/rgb_head_front_center/rgb_image/compressed
```

Run the recorder without `--camera_tap_addr`. It subscribes with keep-last 1,
best-effort QoS by default, stores the original compressed bytes, and keeps the
ROS header timestamp. This removes the extra vision-bridge/TCP hop. The vision
bridge is then needed only for the PICO first-person display, not for dataset
recording.

Both the direct ROS callback and the TCP camera-tap reader use a thread-safe
latest-only camera slot outside the shared joint/IMU/controller FIFO. If disk
writing is temporarily slower than the camera, a new image replaces the one
pending image instead of building an old-frame backlog. Every such replacement
is reported separately as
`ingress_drops.camera_coalesced`; it is intentional low-latency coalescing, not
a DDS or TCP transport gap. The dispatcher preserves receive-time order against
older FIFO events, gives non-camera streams normal FIFO service, and flushes the
last pending camera frame during clean shutdown.

For direct ROS recording, `camera_head.sequence` is assigned at the recorder
callback. It detects callback-to-disk coalescing but cannot prove whether DDS
or the camera publisher dropped an earlier image. Use gaps in the preserved
ROS source timestamp as the upstream-loss diagnostic. The TCP path additionally
has the vision bridge's transport sequence and can distinguish a network gap.

Use the TCP tap when the recorder runs on another computer or cannot discover
the robot-internal camera topic. The tap preserves the same original ROS header
timestamp, so it remains usable for source-time alignment, but transport delay
and latest-only drops are larger quality diagnostics.

For the TCP-tap path, start the deployed vision bridge on the robot normally.
Its default configuration enables the recorder tap on `0.0.0.0:28706`; look
for:

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

## 3. Run the raw recorder

Recommended robot-local form (direct ROS camera):

```bash
source /opt/ros/humble/setup.bash
source /digit/software/x1_mc/install/setup.bash
export LD_LIBRARY_PATH=/digit/software/x1_mc/runtime_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

cd /digit/run/x2_vr_data_collection
/digit/software/x2_vr_bridge/.venv/bin/python x2_vr_recorder.py \
  --sensor_profile aimdk \
  --tap_addr tcp://127.0.0.1:28704 \
  --camera_topic /aima/hal/sensor/rgb_head_front_center/rgb_image/compressed \
  --camera_qos_reliability best_effort \
  --task "touch the red button with the left hand" \
  --output_root /digit/run/Datasets/x2_vr/raw
```

For a laptop-side recorder, source ROS and the built motion-control workspace
before invoking the existing GMR virtualenv:

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
gaps are recorded as `camera_tap_transport` ingress drops. Frames received
successfully but superseded in the recorder's latest-only slot are counted
separately as `camera_coalesced`, so network loss is not conflated with local
low-latency coalescing. Replace
`172.66.88.241` if the robot IP changes.

The default `--sensor_profile aimdk` reads the real robot topics directly:

```text
/aima/hal/joint/{leg,waist,arm,head}/state
/aima/hal/imu/{torso,chest}/state
```

For simulation or an older deployment that publishes the compatibility topics,
pass `--sensor_profile compat` to use `/joint_states/*` and `/imu/*/data`.

Without `--camera_tap_addr`, the legacy direct ROS camera subscription remains
available and defaults to sensor-style `best_effort` QoS. Keep this setting
even when `ros2 topic info -v` reports a reliable publisher: a reliable offer
is compatible with a best-effort request, while the best-effort subscriber
avoids retransmission backpressure in the recording path. Use a reliable
subscriber only after confirming an actual QoS incompatibility and verifying
that it does not build latency.

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

A writer exception or shutdown timeout is fatal: the active episode remains
`interrupted` (or hidden as `.partial` if even final flushing fails), and the
recorder exits non-zero. A received X/stop edge never promotes a writer-failed
episode to `complete`.

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
collecting many demonstrations. Source/header time is now the default and each
25 Hz tick is anchored to one unique camera capture inside the A-to-X active
interval; joint, IMU, hand and reference samples are associated causally with
that camera time. A camera frame is never repeated to fill a missing source
frame. To reproduce the old arrival-time converter exactly for comparison, add
`--time_basis receiver`: non-camera streams are then sampled on the fixed 25 Hz
target time, camera frames may be reused, and reference frames are interpolated.

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
- `sync.timing_ms`: signed camera-minus-grid offset and causal
  reference/hand/joint/IMU ages (5 floats, deliberately outside the policy
  observation namespace);
- `action`: bridge reference `root xyz + quaternion wxyz + q29`, followed by
  left and right grasp fractions (38 floats total);
- `conversion_report.json`: versioned conversion parameters and thresholds,
  per-raw-episode accepted/skipped/synchronization diagnostics, and the exact
  raw-episode-to-output-segment mapping with frame counts plus first/last
  target and camera timestamps;
- one language task string per episode.

When a synchronized tick is missing, the converter splits at that gap instead
of concatenating the remaining frames and speeding up time.  Existing output
directories are never deleted or overwritten.

The report is written only after LeRobot finalization succeeds and is moved
atomically with the dataset. It records `time_basis`, FPS, all freshness
thresholds, camera rotation, segmentation settings, success filtering and the
partial-source-time setting. Within each reported continuous segment,
`target_timestamp_ns` advances by `1/fps`; the stored signed camera-grid offset
in `sync.timing_ms[0]` relates every dataset frame back to its camera anchor.
`--dry_run` prints the same synchronization diagnostics but never writes into
the raw dataset.

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

Synchronization defaults to source/header timestamps mapped into the recorder
monotonic domain. Camera captures anchor the associated state and action;
recorder receive timestamps remain in the raw files as transport-latency
diagnostics. Reference actions use the bridge command/reply time, not the older
GMR lookback/sample-target time. Source mode is strict: a required stream with
an unmappable source time fails conversion instead of being silently mixed or
partially dropped. Use `--time_basis receiver` for legacy arrival-time data.
`--allow_partial_source_time` is only a diagnostic escape hatch: it drops
unmappable events, reports their counts, and should not be used to produce a
training dataset.

Small callback reordering is reported, while a source timestamp regression
larger than 100 ms is treated as a clock reset/corrupt timeline and fails
strict source conversion instead of being silently sorted into the episode.

Source-time alignment assumes the ROS publishers, bridge and recorder share a
synchronized wall clock. Robot-local recording satisfies this naturally. For
a recorder or bridge on another computer, synchronize all hosts with PTP or
chrony and record/check the clock offset before collecting; being on the same
LAN is not sufficient. Without that evidence, use receiver time only for
diagnosis rather than treating the result as training-quality synchronization.
