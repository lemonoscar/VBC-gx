#!/usr/bin/env bash
set -u

cd "$(dirname "$0")"

echo "[GPU-Wait] Waiting for GPU to become available..."

# 等待 GPU 就绪
gpu_ready=0
for attempt in {1..120}; do
  if nvidia-smi &>/dev/null; then
    gpu_ready=1
    echo "[GPU-Wait] GPU is ready!"
    break
  fi
  echo "[GPU-Wait] Attempt $attempt/120: GPU not ready, waiting 5 seconds..."
  sleep 5
done

if [[ $gpu_ready -eq 0 ]]; then
  echo "[GPU-Wait] GPU still not available after 10 minutes. Aborting."
  exit 1
fi

BASE_CMD=(
  python train_multistate.py
  --task Go2X5PickMulti
  --config data/cfg/go2x5_pickmulti.yaml
  --rl_device cuda:0
  --sim_device cuda:0
  --timesteps 60000
  --headless
  --experiment_dir go2x5-pick-multi-teacher
  --wandb_name smoke_go2x5_teacher_v9low
  --roboinfo
  --small_value_set_zero
  --rand_control
  --stop_pick
)

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[train] attempt ${attempt}"

  if [[ ${attempt} -eq 1 ]]; then
    "${BASE_CMD[@]}" --resume
  else
    "${BASE_CMD[@]}" --resume
  fi

  exit_code=$?
  if [[ ${exit_code} -eq 0 ]]; then
    echo "[train] finished normally"
    exit 0
  fi

  echo "[train] crashed with exit code ${exit_code}, restarting in 5s..."
  sleep 5
done
