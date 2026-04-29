#!/usr/bin/env bash
set -u

cd "$(dirname "$0")"

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
  echo "[stable-train] attempt ${attempt}"

  if [[ ${attempt} -eq 1 ]]; then
    "${BASE_CMD[@]}"
  else
    "${BASE_CMD[@]}" --resume
  fi

  exit_code=$?
  if [[ ${exit_code} -eq 0 ]]; then
    echo "[stable-train] finished normally"
    exit 0
  fi

  echo "[stable-train] crashed with exit code ${exit_code}, restarting in 5s with --resume..."
  sleep 5
done
