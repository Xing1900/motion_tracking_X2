#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_groot_n17_recorder.sh "<task description>"

Example:
  ./run_groot_n17_recorder.sh "put the red block into the box"

The wrapper records robot-local GR00T N1.7 data using the standard X2 paths
and endpoints. It waits briefly for the recorder tap (28704), camera tap
(28706), and C++ tracking telemetry (28707) before starting.

Optional environment overrides:
  X2_MC_ROOT                         default: /digit/software/x1_mc
  X2_VR_BRIDGE_ROOT                  default: /digit/software/x2_vr_bridge
  X2_VR_PYTHON                       default: <bridge root>/.venv/bin/python
  X2_VR_OUTPUT_ROOT                  default: /digit/run/Datasets/x2_vr/raw
  X2_VR_HUMAN_HEIGHT                 default: 1.7
  X2_VR_GMR_MAX_ITER                 default: 0
  X2_VR_LOOKBACK_MS                  default: 25.0
  X2_VR_PORT_WAIT_TIMEOUT_S          default: 10
  X2_VR_SKIP_PORT_PREFLIGHT          set to 1 only for diagnostics
EOF
}

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 ]]; then
  echo "error: pass exactly one quoted task description" >&2
  usage >&2
  exit 2
fi

task_description="$1"
if [[ -z "${task_description//[[:space:]]/}" ]]; then
  echo "error: task description must not be empty" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mc_root="${X2_MC_ROOT:-/digit/software/x1_mc}"
bridge_root="${X2_VR_BRIDGE_ROOT:-/digit/software/x2_vr_bridge}"
python_bin="${X2_VR_PYTHON:-${bridge_root}/.venv/bin/python}"
output_root="${X2_VR_OUTPUT_ROOT:-/digit/run/Datasets/x2_vr/raw}"
human_height="${X2_VR_HUMAN_HEIGHT:-1.7}"
gmr_max_iter="${X2_VR_GMR_MAX_ITER:-0}"
lookback_ms="${X2_VR_LOOKBACK_MS:-25.0}"
port_wait_timeout_s="${X2_VR_PORT_WAIT_TIMEOUT_S:-10}"
if [[ ! "${port_wait_timeout_s}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: X2_VR_PORT_WAIT_TIMEOUT_S must be a positive integer" >&2
  exit 2
fi

ros_setup="/opt/ros/humble/setup.bash"
mc_setup="${mc_root}/install/setup.bash"
recorder="${script_dir}/x2_vr_recorder.py"
raw_writer="${script_dir}/raw_episode_writer.py"
camera_client="${script_dir}/camera_tap_client.py"
schema_module="${script_dir}/schema.py"
telemetry_module="${script_dir}/tracking_telemetry.py"
controller_binary="${mc_root}/install/x1_rl_control/lib/x1_rl_control/x1_rl_control_node"
controller_config="${mc_root}/install/x1_rl_control/share/x1_rl_control/cfg/rl/rl_tracking.yaml"
controller_policy="${mc_root}/install/x1_rl_control/share/x1_rl_control/policy/tracking_policy/policy.onnx"
controller_policy_data="${controller_policy}.data"
hand_config="${mc_root}/install/vr_hand_controller/share/vr_hand_controller/config/vr_hand_controller.yaml"
teleop_bridge="${bridge_root}/bridge/xrobot_teleop_to_pose_zmq_server.py"
gmr_config="${bridge_root}/gmr/general_motion_retargeting/ik_configs/xrobot_to_agibot_x2.json"
gmr_runtime="${bridge_root}/gmr/general_motion_retargeting/motion_retarget.py"

required_files=(
  "${ros_setup}"
  "${mc_setup}"
  "${recorder}"
  "${raw_writer}"
  "${camera_client}"
  "${schema_module}"
  "${telemetry_module}"
  "${controller_binary}"
  "${controller_config}"
  "${controller_policy}"
  "${controller_policy_data}"
  "${hand_config}"
  "${teleop_bridge}"
  "${gmr_config}"
  "${gmr_runtime}"
)
for required_file in "${required_files[@]}"; do
  if [[ ! -f "${required_file}" || ! -r "${required_file}" ]]; then
    echo "error: required runtime file is missing or unreadable: ${required_file}" >&2
    exit 1
  fi
done
if [[ ! -x "${python_bin}" ]]; then
  echo "error: recorder Python is not executable: ${python_bin}" >&2
  exit 1
fi
if [[ ! -x "${controller_binary}" ]]; then
  echo "error: controller binary is not executable: ${controller_binary}" >&2
  exit 1
fi

if existing_recorder_pids="$(pgrep -f -- "${recorder}" || true)"; \
  [[ -n "${existing_recorder_pids}" ]]; then
  echo "error: an X2 VR recorder is already running (PID(s): ${existing_recorder_pids//$'\n'/,})" >&2
  exit 1
fi

if [[ "${X2_VR_SKIP_PORT_PREFLIGHT:-0}" != "1" ]]; then
  if ! command -v ss >/dev/null 2>&1; then
    echo "error: 'ss' is required for endpoint preflight" >&2
    exit 1
  fi

  wait_for_port() {
    local name="$1"
    local port="$2"
    local deadline=$((SECONDS + port_wait_timeout_s))
    while ! ss -H -ltn "sport = :${port}" 2>/dev/null | grep -q .; do
      if (( SECONDS >= deadline )); then
        echo "error: ${name} is not listening on tcp://127.0.0.1:${port}" >&2
        return 1
      fi
      sleep 0.2
    done
  }

  wait_for_port "VR recorder tap" 28704
  wait_for_port "camera tap" 28706
  wait_for_port "tracking telemetry" 28707
fi

set +u
source "${ros_setup}"
source "${mc_setup}"
set -u

export LD_LIBRARY_PATH="${mc_root}/runtime_lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

if ! ros_topics="$(timeout 10 ros2 topic list 2>/dev/null)"; then
  echo "error: timed out while discovering ROS topics" >&2
  exit 1
fi
if ! grep -Fxq '/vr_hand_controller/status' <<<"${ros_topics}"; then
  echo "error: ROS topic /vr_hand_controller/status is not available" >&2
  exit 1
fi

mkdir -p "${output_root}"
if [[ ! -d "${output_root}" || ! -w "${output_root}" ]]; then
  echo "error: output directory is not writable: ${output_root}" >&2
  exit 1
fi

echo "[groot-recorder] task: ${task_description}"
echo "[groot-recorder] output: ${output_root}"

exec "${python_bin}" "${recorder}" \
  --record_profile groot_n17 \
  --sensor_profile aimdk \
  --tap_addr tcp://127.0.0.1:28704 \
  --tracking_tap_addr tcp://127.0.0.1:28707 \
  --camera_tap_addr tcp://127.0.0.1:28706 \
  --hand_status_qos_depth 1 \
  --hand_status_qos_reliability best_effort \
  --bridge_actual_human_height "${human_height}" \
  --bridge_gmr_max_iter "${gmr_max_iter}" \
  --bridge_lookback_ms "${lookback_ms}" \
  --bridge_min_link_height 0.0 \
  --bridge_min_link_height_align_strategy startup_fixed \
  --bridge_min_link_height_bootstrap_frames 10 \
  --provenance_file "controller_binary=${controller_binary}" \
  --provenance_file "controller_config=${controller_config}" \
  --provenance_file "controller_policy=${controller_policy}" \
  --provenance_file "controller_policy_data=${controller_policy_data}" \
  --provenance_file "hand_config=${hand_config}" \
  --provenance_file "teleop_bridge=${teleop_bridge}" \
  --provenance_file "gmr_config=${gmr_config}" \
  --provenance_file "gmr_runtime=${gmr_runtime}" \
  --task "${task_description}" \
  --output_root "${output_root}"
