#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

: "${LOW_POLICY_PATH:?Set LOW_POLICY_PATH to the absolute schema-v2 12D low-level checkpoint}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_NAME="${TRAIN_NAME:-go2x5_teacher_v11_cooperative}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-go2x5-pick-multi-teacher}"
NUM_ENVS="${NUM_ENVS:-256}"
TIMESTEPS="${TIMESTEPS:-60000}"
SEED="${SEED:-43}"
RL_DEVICE="${RL_DEVICE:-cuda:0}"
SIM_DEVICE="${SIM_DEVICE:-cuda:0}"
GRAPHICS_DEVICE_ID="${GRAPHICS_DEVICE_ID:-0}"

if [[ ! -f "${LOW_POLICY_PATH}" ]]; then
  echo "Low-level checkpoint not found: ${LOW_POLICY_PATH}" >&2
  exit 2
fi

"${PYTHON_BIN}" train_multistate.py \
  --task Go2X5PickMulti \
  --config data/cfg/go2x5_pickmulti.yaml \
  --low_policy_path "${LOW_POLICY_PATH}" \
  --rl_device "${RL_DEVICE}" \
  --sim_device "${SIM_DEVICE}" \
  --graphics_device_id "${GRAPHICS_DEVICE_ID}" \
  --timesteps "${TIMESTEPS}" \
  --num_envs "${NUM_ENVS}" \
  --seed "${SEED}" \
  --headless \
  --experiment_dir "${EXPERIMENT_DIR}" \
  --wandb_name "${TRAIN_NAME}" \
  --roboinfo \
  --small_value_set_zero \
  --rand_control \
  --stop_pick \
  "$@"
