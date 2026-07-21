#!/usr/bin/env python3
"""Create an untrained 12D Go2-X5 checkpoint for production-loader parity."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "third_party/rsl_rl"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rsl_rl.modules import ActorCritic  # noqa: E402
from tools.go2x5_runtime_parity import (  # noqa: E402
    SMOKE_HISTORY_LEN,
    SMOKE_NUM_PRIV,
    SMOKE_NUM_PROP,
    build_smoke_actor_critic,
    canonical_json_sha256,
    smoke_actor_critic_kwargs,
    validate_schema_v2_checkpoint,
)


CONTRACT_PROFILE = "simple_deployment_smoke_v1"
PURPOSE = "runtime_parity_smoke_only"


def actor_critic_kwargs():
    return smoke_actor_critic_kwargs()


def build_model(seed: int = 20260713):
    return build_smoke_actor_critic(ActorCritic, seed)


def build_metadata(cfg, asset_path: Path, seed: int):
    env_cfg = cfg["env"]
    contract = env_cfg["lowPolicyContract"]
    alignment = {
        "schema_version": 2,
        "asset_file": str(asset_path),
        "asset_sha256": hashlib.sha256(asset_path.read_bytes()).hexdigest(),
        "action_dim": 12,
        "num_arm_actions": 0,
        "policy_output_tanh": True,
        "num_torques": 12,
        "num_gripper_joints": 2,
        "num_proprio": SMOKE_NUM_PROP,
        "num_priv": SMOKE_NUM_PRIV,
        "history_len": SMOKE_HISTORY_LEN,
        "num_observations": SMOKE_NUM_PROP * (SMOKE_HISTORY_LEN + 1) + SMOKE_NUM_PRIV,
        "observe_gait_commands": False,
        "reorder_dofs": True,
        "ee_body_name": env_cfg["eeBodyName"],
        "arm_base_offset": env_cfg["armBaseOffset"],
        "control_contract": contract,
        "control_contract_sha256": canonical_json_sha256(contract),
        "contract_profile": CONTRACT_PROFILE,
        "purpose": "parity_smoke",
        "trained": False,
        "random_seed": seed,
        "curriculum": {"enabled": False, "profile_name": CONTRACT_PROFILE},
    }
    return {"purpose": PURPOSE, "trained": False, "go2x5_alignment": alignment}


def create_checkpoint(output: Path, seed: int = 20260713):
    cfg_path = ROOT / "high-level/data/cfg/go2x5_pickmulti.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    asset_path = ROOT / "low-level/resources/robots/go2x5/go2_x5.urdf"
    model = build_model(seed)
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter": 0,
        "infos": {"purpose": PURPOSE, "trained": False},
        "metadata": build_metadata(cfg, asset_path, seed),
    }
    validate_schema_v2_checkpoint(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/go2x5_schema_v2_smoke.pt"))
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--contract-profile", choices=[CONTRACT_PROFILE], default=CONTRACT_PROFILE)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = create_checkpoint(args.output, seed=args.seed)
    alignment = checkpoint["metadata"]["go2x5_alignment"]
    print(f"wrote untrained 12D parity smoke checkpoint: {args.output}")
    print(f"control_contract_sha256={alignment['control_contract_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
