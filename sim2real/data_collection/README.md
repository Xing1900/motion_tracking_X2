# X2 VR demonstration recorder

This directory contains the data path that sits beside, not inside, the X2
real-time controller:

```text
PICO/XRoboToolkit -> GMR teleop bridge -> rl_tracking
                         |                 |
        controller PUB tap :28704         | atomic tracking telemetry :28707
                         v                 v
                        x2_vr_recorder.py
                         ^               ^
       bounded camera TCP tap :28706      | ROS 2 hand status only
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

For GR00T N1.7 collection, run with `--record_profile groot_n17`. It is the
lowest-load profile and stores exactly four logical streams:

- `camera_head`: original compressed X2 first-person images through the
  independent bounded camera dispatcher/writer;
- `tracking_telemetry`: one atomic 25 Hz snapshot made by the C++ controller:
  measured q29/dq29, policy-consumed root/q29 reference, root angular velocity,
  projected gravity, raw policy output and final q29 command;
- `hand_command`: latest accepted left/right grasp fractions and `active`
  validity from `/vr_hand_controller/status`;
- `controller`: the A/X edges that delimit each episode.

This profile does **not** subscribe to joint/IMU ROS topics and does not receive
the bridge's `reference`, `xr` or `retarget` payloads. That avoids duplicating
state transport and makes the policy-consumed C++ reference—not an earlier GMR
message—the training label.

The older `--record_profile vla` remains available for comparison and stores:

- `camera_head`: original compressed X2 first-person images;
- `joint_states`: the real 29-DoF positions, velocities, effort and faults,
  normalized from AimDK `JointStateArray` messages;
- `imu_torso`: base orientation, angular velocity and acceleration;
- `reference`: bridge output `root xyz + quaternion wxyz + q29`, including
  `retarget_age_ms` for latency screening;
- `hand_command`: the high-level command accepted by the hand controller:
  left/right grasp fractions, `active` validity and sequence (not all OmniHand
  joint targets);
- `controller`: 50 Hz PICO state, including the A/X edges that delimit each
  episode.

The default `full` profile additionally keeps the following diagnostic streams:

- `xr`: 24 raw XR body poses, headset pose and controller state.
- `retarget`: the latest raw GMR `root xyz + quaternion wxyz + q29` result.
- head joint state and `imu_chest`, in addition to the VLA state streams.

The 29-joint action order is copied from `rl_tracking.yaml/BaseConfig.seq`:
legs, waist `yaw/pitch/roll`, then each arm with wrist `yaw/pitch/roll`.

## 1. Restart the VR bridge with the recorder tap

The bridge tap is disabled during normal teleoperation so it adds no
serialization load to the latency-sensitive path. For a `groot_n17` session,
publish only controller boundaries:

```bash
--tap_bind_addr tcp://*:28704 \
--tap_streams controller
```

The C++ tracking controller separately publishes the atomic state/reference
snapshot from a background, non-blocking worker. Its YAML settings are:

```yaml
tracking_telemetry_enable: true
tracking_telemetry_rate_hz: 25.0
tracking_telemetry_pub_addr: tcp://127.0.0.1:28707
```

At startup confirm the configured telemetry line plus:

```text
[tracking_telemetry] publishing at tcp://127.0.0.1:28707
```

and verify the listener with `ss -ltnp | grep 28707`. The controller thread
only copies one fixed-size sample into a bounded queue; JSON/ZMQ work happens
off the real-time path, and overload is represented by sequence gaps instead
of control-loop blocking.

For the older ROS-state `vla` session, add:

```bash
--tap_bind_addr tcp://*:28704 \
--tap_streams controller reference
```

to the normal X2 bridge command. After restarting, it should print:

```text
[teleop_tap] publishing recorder events at tcp://*:28704
```

This prevents the bridge from constructing and serializing the large raw XR
and intermediate retarget tap payloads. Omit `--tap_streams` (or pass `'*'`) for
the full diagnostic recording profile.

The bridge queues tap events without blocking control. Every new bridge event
keeps the legacy global `tap_seq` and also carries an independent
`tap_topic_seq`. The recorder checks the latter separately for every received
stream in both VLA and full profiles. A loss is reported clearly as, for
example, `ingress_drops.teleop_tap_transport.reference`; intentionally filtered
XR/retarget traffic therefore never looks like a controller/reference gap.
With an older bridge that has no `tap_topic_seq`, full recording falls back to
the global `tap_seq`; filtered VLA recording stays compatible but cannot prove
per-topic tap continuity until the bridge is updated.

## 2. Choose the camera path and check robot sensors

For sustained collection on the robot, prefer the loopback TCP tap:

```text
tcp://127.0.0.1:28706
```

The vision bridge remains the only large ROS camera subscriber and forwards the
unchanged compressed bytes plus their original ROS header timestamp. Its sender
and the recorder camera reader run outside the recorder's joint/IMU ROS
executor. On the same robot, loopback TCP adds negligible network uncertainty
and avoids making the recorder deserialize the high-resolution image topic in
the same process as hand and state callbacks.

Both the direct ROS callback and the TCP camera-tap reader now feed a dedicated
bounded camera ingress dispatcher. Camera frames never enter the public
joint/IMU/reference FIFO, and the camera dispatcher only routes each frame into
the active episode's asynchronous JPEG writer. A/X and pre/post-roll boundaries
remain timestamp based, including when the independent camera worker reaches a
frame before the public dispatcher reaches an older A edge. If either bounded
stage cannot keep up, it drops the oldest pending image instead of building a
delayed video: `camera_ingress_coalesced` identifies pressure before episode
routing and `camera_writer_coalesced` identifies pressure in the disk writer.
Neither is a DDS/TCP transport gap. New recordings should no longer report the
old shared-dispatcher counter `camera_coalesced`. Frames abandoned after an
actual disk-writer exception are reported separately as
`camera_writer_after_fatal`.

The old recorder before commit `5d69c50` could report nearly 30 stored images
per second because every JPEG entered the shared 8192-event FIFO. When writing
fell behind, that number represented images eventually flushed seconds later,
not a live 30 Hz path. The bounded design deliberately prevents that failure;
the independent writer restores throughput without restoring an unbounded old-
frame backlog.

For direct ROS recording, `camera_head.sequence` is assigned at the recorder
callback. It detects callback-to-disk coalescing but cannot prove whether DDS
or the camera publisher dropped an earlier image. Use gaps in the preserved
ROS source timestamp as the upstream-loss diagnostic. The TCP path additionally
has the vision bridge's transport sequence and can distinguish a network gap.

Direct ROS camera subscription remains available as a diagnostic fallback by
omitting `--camera_tap_addr`. It uses keep-last 1, best-effort QoS and preserves
the same source timestamp, but on the current robot it competes with the other
Python ROS callbacks and is not the recommended sustained-collection path.

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

Recommended robot-local GR00T N1.7 form (all high-rate paths stay on loopback):

```bash
source /opt/ros/humble/setup.bash
source /digit/software/x1_mc/install/setup.bash
export LD_LIBRARY_PATH=/digit/software/x1_mc/runtime_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

cd /digit/run/x2_vr_data_collection
/digit/software/x2_vr_bridge/.venv/bin/python x2_vr_recorder.py \
  --record_profile groot_n17 \
  --sensor_profile aimdk \
  --tap_addr tcp://127.0.0.1:28704 \
  --tracking_tap_addr tcp://127.0.0.1:28707 \
  --camera_tap_addr tcp://127.0.0.1:28706 \
  --bridge_actual_human_height 1.7 \
  --bridge_gmr_max_iter 0 \
  --bridge_lookback_ms 25.0 \
  --bridge_min_link_height 0.0 \
  --bridge_min_link_height_align_strategy startup_fixed \
  --bridge_min_link_height_bootstrap_frames 10 \
  --provenance_file controller_binary=/digit/software/x1_mc/install/x1_rl_control/lib/x1_rl_control/x1_rl_control_node \
  --provenance_file controller_config=/digit/software/x1_mc/install/x1_rl_control/share/x1_rl_control/cfg/rl/rl_tracking.yaml \
  --provenance_file controller_policy=/digit/software/x1_mc/install/x1_rl_control/share/x1_rl_control/policy/tracking_policy/policy.onnx \
  --provenance_file controller_policy_data=/digit/software/x1_mc/install/x1_rl_control/share/x1_rl_control/policy/tracking_policy/policy.onnx.data \
  --provenance_file hand_config=/digit/software/x1_mc/install/vr_hand_controller/share/vr_hand_controller/config/vr_hand_controller.yaml \
  --provenance_file teleop_bridge=/actual/path/to/xrobot_teleop_to_pose_zmq_server.py \
  --provenance_file gmr_config=/actual/path/to/xrobot_to_agibot_x2.json \
  --provenance_file gmr_runtime=/actual/path/to/motion_retarget.py \
  --task "touch the red button with the left hand" \
  --output_root /digit/run/Datasets/x2_vr/raw
```

Replace the three `/actual/path/...` entries with the files used by the running
robot processes (use `readlink -f` if they are symlinks). Set every
`--bridge_*` value to the effective value used to start that bridge; the values
above match the current robot bundle (`1.7`, `0`, and `25 ms`), not an
instruction to overwrite future robot-specific settings. The recorder
automatically hashes its running script plus the eight
declared artifacts into every manifest. The strict N1.7 converter rejects a
missing hash, missing bridge runtime value, or any batch that mixes different
recorder, controller, policy, bridge/GMR/hand files or effective retarget
parameters; this also catches uncommitted files edited directly on robot 241.

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
  --record_profile groot_n17 \
  --tap_addr tcp://172.66.88.241:28704 \
  --tracking_tap_addr tcp://172.66.88.241:28707 \
  --camera_tap_addr tcp://172.66.88.241:28706 \
  --bridge_actual_human_height 1.7 \
  --bridge_gmr_max_iter 0 \
  --bridge_lookback_ms 25.0 \
  --bridge_min_link_height 0.0 \
  --bridge_min_link_height_align_strategy startup_fixed \
  --bridge_min_link_height_bootstrap_frames 10 \
  --provenance_file controller_binary=/local/copy/of/x1_rl_control_node \
  --provenance_file controller_config=/local/copy/of/rl_tracking.yaml \
  --provenance_file controller_policy=/local/copy/of/policy.onnx \
  --provenance_file controller_policy_data=/local/copy/of/policy.onnx.data \
  --provenance_file hand_config=/local/copy/of/vr_hand_controller.yaml \
  --provenance_file teleop_bridge=/local/copy/of/xrobot_teleop_to_pose_zmq_server.py \
  --provenance_file gmr_config=/local/copy/of/xrobot_to_agibot_x2.json \
  --provenance_file gmr_runtime=/local/copy/of/motion_retarget.py \
  --task "touch the red button with the left hand" \
  --output_root ~/Datasets/x2_vr/raw
```

The remote telemetry example requires the controller bind address to be
`tcp://*:28707` (and an appropriate firewall rule). Keep the default loopback
bind and record on the robot whenever possible; it removes inter-host clock and
network uncertainty from the training path.

With `--camera_tap_addr` enabled, the recorder automatically skips its ROS
camera subscription. In `groot_n17`, joint and IMU subscriptions are also
disabled; only the latest-state hand status remains in the ROS executor. The
camera client reconnects after a bridge/network interruption, and sequence
gaps are recorded as `camera_tap_transport` ingress drops. Frames received
successfully but superseded in the independent bounded camera ingress queue are
counted separately as `camera_ingress_coalesced`, so network loss is not
conflated with local pressure. Frames superseded in the bounded disk queue are
counted as `camera_writer_coalesced`. Replace
`172.66.88.241` if the robot IP changes.

With the older `vla` record profile, `--sensor_profile aimdk` reads only the state
topics used by the final dataset:

```text
/aima/hal/joint/{leg,waist,arm}/state
/aima/hal/imu/torso/state
```

Use `--record_profile full` when diagnosing XR/GMR or head/chest sensors. The
raw schema and converter are unchanged; the diagnostic JSONL files are simply
absent in VLA episodes. In VLA mode the ZeroMQ subscriber requests only
`controller` and `reference`, so it also avoids decoding XR/retarget JSON. The
bridge emits a `tap_topic_seq` for each stream, so controller and reference
transport gaps are checked independently even when XR/retarget are intentionally
filtered. The legacy global `tap_seq` is used only as an old-bridge fallback in
`full`. Required-stream timing gaps additionally validate controller and
reference continuity in every converted episode.

Hand status uses `best_effort`, keep-last depth 1 by default. It is a
latest-state signal: old heartbeats are discarded rather than replayed after a
callback stall. Override with `--hand_status_qos_reliability reliable` only for
diagnosis. The manifest records
`hand_status_delivery_semantics=latest_state`, reliability and depth so offline
conversion can treat sequence gaps as delivery diagnostics while still
rejecting stale or inactive hand labels.

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
`hand_command`, `camera_head`, and `tracking_telemetry` counts in the
`groot_n17` profile. The ROS-state `vla` profile instead expects `joint_states`
and `imu_torso`.
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

Make the first trial only 5--10 seconds. In `groot_n17`, confirm
`tracking_telemetry` is close to 25 Hz, `camera_head` is close to the physical
camera rate, the public/camera queues remain low, and sequence/drop counters do
not grow. Then press X and immediately run the synchronization dry-run below.

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
    │   ├── tracking_telemetry.jsonl # groot_n17 profile
    │   ├── imu_torso.jsonl
    │   ├── joint_states.jsonl
    │   ├── reference.jsonl
    │   ├── retarget.jsonl  # full profile only
    │   └── xr.jsonl        # full profile only
    └── images/head_rgb/*.jpg
```

The recorder does not assume that pressing stop means success.  Annotate an
episode after reviewing it:

```bash
~/Documents/gmr/.venv/bin/python annotate_episode.py \
  ~/Datasets/x2_vr/raw/episode_000000 --success true
```

## 4. Validate GR00T N1.7 synchronization

For `groot_n17` raw episodes, run the strict source-time converter as a dry run:

```bash
~/Documents/gmr/.venv/bin/python convert_to_groot_n17.py \
  --raw_root ~/Datasets/x2_vr/raw \
  --fps 25 \
  --dry_run
```

It anchors each 25 Hz row to one unique camera frame, then causally takes the
newest atomic C++ telemetry and latest active hand state at or before that
exposure. Controller-generated transition and padded references are rejected;
a fresh bridge fallback remains usable only while its consumed-reference
source age is at most 80 ms. Camera/telemetry/hand age, reference source age,
sequence gaps, skipped reasons and q-reference/command tracking RMSE are kept
in the conversion report. Any rejected tick splits continuity, and fragments
shorter than the 40-step (1.6 s) action horizon are not emitted.

The output row is 104-D state and 40-D consumed reference action. Raw telemetry
keeps the global aligned reference; each output segment is rigidly rebased into
its first consumed-reference frame, so stored poses are absolute within that
episode-local frame.
GR00T's modality processor converts only the root and joint reference groups
to the configured local/relative representation; do not pre-difference them in
this converter.

This 104-D schema assumes the two X2 head joints remain fixed during a
demonstration. If another controller moves the head camera, add head yaw/pitch
to the synchronous controller telemetry and state schema before collecting the
training set; otherwise identical images cannot be interpreted under a known
camera extrinsic.

### Legacy `vla`/`full` synchronization

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
python convert_to_groot_n17.py \
  --raw_root ~/Datasets/x2_vr/raw \
  --output_root ~/Datasets/x2_vr/x2_groot_n17_v3 \
  --repo_id local/x2_groot_n17 \
  --modality_json ~/Documents/Isaac-GR00T/examples/X2/modality.json \
  --min_segment_frames 40 \
  --require_success \
  --fps 25
```

This writes the 104-D state, 40-D episode-local absolute consumed-reference action, four
synchronization diagnostics and a `conversion_report.json` containing skipped
reasons, exact raw-to-output segment provenance, the shared capture contract
(including recorder/artifact hashes and bridge effective parameters), and
tracking RMSE. It also copies the X2 `modality.json` into `meta/`.

GR00T N1.7 currently consumes its LeRobot-v2 flavor. Convert the resulting v3
dataset with the helper in
`Isaac-GR00T/scripts/lerobot_conversion/convert_v3_to_v2.py`; this checkout's
helper preserves `meta/modality.json` and `conversion_report.json`. Regenerate
dataset statistics before fine-tuning with
`Isaac-GR00T/examples/X2/x2_reference_config.py`.

```bash
cd ~/Documents/Isaac-GR00T/scripts/lerobot_conversion
python convert_v3_to_v2.py \
  --root ~/Datasets/x2_vr \
  --repo-id x2_groot_n17_v3
```

For backward-compatible 68-D/38-D `vla`/`full` episodes, replace the command
above with `convert_to_lerobot.py` and use a separate output directory.

The X2 head camera is mounted upside down for this stream, so conversion rotates
frames by 180 degrees by default. Raw compressed images remain unchanged. Pass
`--camera_rotation_deg 0` if the camera publisher has already corrected them.

Use `--require_success` after the episodes have been annotated.  The legacy converter
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
For new manifests marked `hand_status_delivery_semantics=latest_state`, status
sequence gaps are reported for QA but are not themselves evidence that the
current grasp label is wrong; freshness and `active` remain mandatory. Legacy
recordings without that semantic retain the stricter sequence-gap rejection.
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
