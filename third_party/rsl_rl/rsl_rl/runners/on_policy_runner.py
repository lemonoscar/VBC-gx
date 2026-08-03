# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
import hashlib
from collections import deque
import statistics
from numbers import Number

import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv

import wandb
from torchinfo import summary

class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( self.env.cfg.env.num_proprio,
                                                        self.env.cfg.env.num_proprio,
                                                        self.env.num_actions,
                                                        **self.policy_cfg, 
                                                        num_priv=env.cfg.env.num_priv,
                                                        num_hist=env.cfg.env.history_len, 
                                                        num_prop=env.cfg.env.num_proprio,
                                                        ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        summary(self.alg.actor_critic)

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tensorboard_log_dir = os.path.join(log_dir, "tensorboard") if log_dir is not None else None
        self._tensorboard_warning_printed = False
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.warm_start_provenance = None
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]

        _, _ = self.env.reset()

        if self.alg.actor_critic.num_arm_actions > 0:
            self.alg.set_arm_default_coeffs(self.env.p_gains[12:], self.env.d_gains[12:], self.env.default_dof_pos[-7:-2])
        
    def set_it(self, it):
        self.current_learning_iteration = it
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # init metrics
        mean_value_loss = 0.
        mean_surrogate_loss = 0.
        mean_arm_torques_loss = 0.
        value_mixing_ratio = 0.
        torque_supervision_weight = 0.
        mean_hist_latent_loss = 0.
        mean_priv_reg_loss = 0.
        priv_reg_coef = 0.

        # initialize writer
        if self.log_dir is not None and self.writer is None:
            if SummaryWriter is None:
                if not self._tensorboard_warning_printed:
                    print("TensorBoard is unavailable: install tensorboard to enable local event logging.")
                    self._tensorboard_warning_printed = True
            else:
                os.makedirs(self.tensorboard_log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=self.tensorboard_log_dir, flush_secs=10)
                print(f"TensorBoard logging to: {self.tensorboard_log_dir}")
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        armrewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        donebuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_arm_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            # self.env.update_command_curriculum()
            if hasattr(self.env, "set_training_iteration"):
                self.env.set_training_iteration(it)

            start = time.time()
            hist_encoding = it % self.dagger_update_freq == 0

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, hist_encoding)
                    obs, privileged_obs, rewards, arm_rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, arm_rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), arm_rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, arm_rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_arm_reward_sum += arm_rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        armrewbuffer.extend(cur_arm_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        donebuffer.append(len(new_ids) / self.env.num_envs)
                        cur_reward_sum[new_ids] = 0
                        cur_arm_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            # self.alg.storage.clear()
            
            # mean_value_loss, mean_surrogate_loss, mean_arm_torques_loss, value_mixing_ratio, torque_supervision_weight, mean_priv_reg_loss, priv_reg_coef = self.alg.update()
            if hist_encoding:
                mean_hist_latent_loss = self.alg.update_dagger()
            else:
                mean_value_loss, mean_surrogate_loss, mean_arm_torques_loss, value_mixing_ratio, torque_supervision_weight, mean_priv_reg_loss, priv_reg_coef = self.alg.update()
            
            stop = time.time()
            learn_time = stop - start
            metric_snapshot = {}
            if self.log_dir is not None:
                metric_snapshot = self.log(locals())
            completed_iterations = it + 1
            if hasattr(self.env, "update_auto_curriculum"):
                self.env.update_auto_curriculum(completed_iterations, metric_snapshot)
            if completed_iterations % self.save_interval == 0:
                self.save(
                    os.path.join(self.log_dir, 'model_{}.pt'.format(completed_iterations)),
                    completed_iterations,
                )
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)), self.current_learning_iteration)
        if self.writer is not None:
            self.writer.flush()

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # wandb.log({'Episode/' + key: value}, step=locs['it'])
                if "rew" in key:
                    wandb_dict['Episode_rew/' + key] = value
                elif "metric" in key:
                    wandb_dict['Episode_metric/' + key] = value
                else:
                    wandb_dict['Episode/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        leg_mean_std = self.alg.actor_critic.std[:, :12].mean()
        if self.alg.actor_critic.std.shape[1] > 12:
            arm_mean_std = self.alg.actor_critic.std[:, 12:].mean()
        else:
            arm_mean_std = leg_mean_std.new_tensor(0.)
        std_numpy = self.alg.actor_critic.std.cpu().detach().numpy()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        wandb_dict['Loss/value_function'] = locs['mean_value_loss']
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        wandb_dict['Loss/hist_latent_loss'] = locs['mean_hist_latent_loss']
        wandb_dict['Loss/priv_reg_loss'] = locs['mean_priv_reg_loss']
        wandb_dict['Loss/priv_ref_lambda'] = locs['priv_reg_coef']
        wandb_dict['Loss/arm_torques_loss'] = locs['mean_arm_torques_loss']
        wandb_dict['Loss/value_mixing_ratio'] = locs['value_mixing_ratio']
        wandb_dict['Loss/torque_supervision_weight'] = locs['torque_supervision_weight']
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
        wandb_dict['Policy/leg_mean_noise_std'] = leg_mean_std.item()
        wandb_dict['Policy/arm_mean_noise_std'] = arm_mean_std.item()
        wandb_dict['Policy/noise_std_dist'] = wandb.Histogram(std_numpy)
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if hasattr(self.env, "get_curriculum_log_info"):
            wandb_dict.update(self.env.get_curriculum_log_info())
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_arm_reward'] = statistics.mean(locs['armrewbuffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
            wandb_dict['Train/dones'] = statistics.mean(locs['donebuffer'])
            # wandb.log({'Train/mean_reward/time': statistics.mean(locs['rewbuffer'])}, step=self.tot_time)
            # wandb.log({'Train/mean_episode_length/time': statistics.mean(locs['lenbuffer'])}, step=self.tot_time)

        self._log_tensorboard(wandb_dict, locs['it'])
        wandb.log(wandb_dict, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'History latent supervision loss:':>{pad}} {locs['mean_hist_latent_loss']:.4f}\n"""
                          f"""{'Privileged info regularizer loss:':>{pad}} {locs['mean_priv_reg_loss']:.4f}\n"""
                          f"""{'Privileged info regularizer lambda:':>{pad}} {locs['priv_reg_coef']:.4f}\n"""
                          f"""{'Leg mean action noise std:':>{pad}} {leg_mean_std.item():.2f}\n"""
                          f"""{'Arm mean action noise std:':>{pad}} {arm_mean_std.item():.2f}\n"""
                          f"""{'action noise std distribution:':>{pad}} {std_numpy.tolist()}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Dones:':>{pad}} {statistics.mean(locs['donebuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'History latent supervision loss:':>{pad}} {locs['mean_hist_latent_loss']:.4f}\n"""
                          f"""{'Leg mean action noise std:':>{pad}} {leg_mean_std.item():.2f}\n"""
                          f"""{'Arm mean action noise std:':>{pad}} {arm_mean_std.item():.2f}\n"""
                          f"""{'action noise std distribution:':>{pad}} {std_numpy.tolist()}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * max(
                               locs['tot_iter'] - locs['it'] - 1, 0):.1f}s\n""")
        print(log_string)
        return wandb_dict

    @staticmethod
    def _to_tensorboard_scalar(value):
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            return value.detach().cpu().item()
        if isinstance(value, Number):
            return float(value)
        return None

    def _log_tensorboard(self, metrics, step):
        if self.writer is None:
            return
        for key, value in metrics.items():
            if key == "Policy/noise_std_dist":
                continue
            scalar = self._to_tensorboard_scalar(value)
            if scalar is not None:
                self.writer.add_scalar(key, scalar, step)
        self.writer.add_scalar("Perf/total_timesteps", self.tot_timesteps, step)
        self.writer.add_scalar("Perf/total_time", self.tot_time, step)
        self.writer.add_histogram("Policy/noise_std_dist", self.alg.actor_critic.std.detach().cpu(), step)
        self.writer.flush()

    def save(self, path, it, infos=None):
        metadata = self.env.get_training_metadata() if hasattr(self.env, "get_training_metadata") else None
        if self.warm_start_provenance is not None:
            metadata = dict(metadata or {})
            metadata["warm_start"] = dict(self.warm_start_provenance)
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'hist_encoder_optimizer_state_dict': self.alg.hist_encoder_optimizer.state_dict(),
            'algorithm_counter': self.alg.counter,
            'runner_state': {
                'tot_timesteps': self.tot_timesteps,
                'tot_time': self.tot_time,
            },
            'iter': it,
            'infos': infos,
            'metadata': metadata,
            }, path)

    @staticmethod
    def _checkpoint_sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def warm_start(self, path):
        """Load compatible network weights into a genuinely fresh training run.

        This deliberately does not restore PPO/history-optimizer state, policy
        exploration standard deviation, counters, elapsed time, or environment
        curriculum state.  It is not a relaxed form of ``load()``.
        """
        if self.current_learning_iteration != 0 or self.alg.counter != 0:
            raise RuntimeError("weights-only warm-start must run before the first learning iteration")
        if self.tot_timesteps != 0 or self.tot_time != 0:
            raise RuntimeError("weights-only warm-start requires an unused runner")
        if self.alg.optimizer.state or self.alg.hist_encoder_optimizer.state:
            raise RuntimeError("weights-only warm-start requires fresh optimizer state")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Warm-start checkpoint does not exist: {path}")

        loaded_dict = torch.load(path, map_location=self.device)
        if not isinstance(loaded_dict, dict):
            raise RuntimeError("Warm-start checkpoint root must be a dictionary")
        source_state = loaded_dict.get("model_state_dict")
        if not isinstance(source_state, dict):
            raise RuntimeError("Warm-start checkpoint has no model_state_dict")
        metadata = loaded_dict.get("metadata")
        require_metadata = bool(
            getattr(getattr(self.env.cfg, "env", None), "require_training_metadata", False)
        )
        compatibility = {}
        if hasattr(self.env, "validate_warm_start_metadata"):
            compatibility = self.env.validate_warm_start_metadata(metadata, path)
        elif require_metadata:
            raise RuntimeError(
                "This environment requires checkpoint metadata but has no warm-start validator"
            )

        target_state = self.alg.actor_critic.state_dict()
        source_keys = set(source_state)
        target_keys = set(target_state)
        if source_keys != target_keys:
            missing = sorted(target_keys - source_keys)
            unexpected = sorted(source_keys - target_keys)
            raise RuntimeError(
                "Warm-start model keys do not match: "
                f"missing={missing}, unexpected={unexpected}"
            )

        for name, target_tensor in target_state.items():
            source_tensor = source_state[name]
            if not isinstance(source_tensor, torch.Tensor):
                raise RuntimeError(f"Warm-start model value is not a tensor: {name}")
            if source_tensor.shape != target_tensor.shape:
                raise RuntimeError(
                    f"Warm-start tensor shape mismatch for {name}: "
                    f"checkpoint={tuple(source_tensor.shape)}, current={tuple(target_tensor.shape)}"
                )
            if source_tensor.dtype != target_tensor.dtype:
                raise RuntimeError(
                    f"Warm-start tensor dtype mismatch for {name}: "
                    f"checkpoint={source_tensor.dtype}, current={target_tensor.dtype}"
                )
            if (source_tensor.is_floating_point() or source_tensor.is_complex()) and not bool(
                torch.all(torch.isfinite(source_tensor))
            ):
                first = torch.nonzero(~torch.isfinite(source_tensor), as_tuple=False)[0].tolist()
                raise FloatingPointError(
                    f"Non-finite warm-start tensor {name} at index {first}"
                )

        # The old policy weights are useful, but the new task must begin with
        # its reviewed exploration schedule instead of inheriting a late-run std.
        preserved_parameters = ["std"]
        candidate_state = dict(source_state)
        for name in preserved_parameters:
            if name not in target_state:
                raise RuntimeError(f"Warm-start preservation key is absent from the model: {name}")
            candidate_state[name] = target_state[name].detach().clone()
        self.alg.actor_critic.load_state_dict(candidate_state, strict=True)

        source_iteration = loaded_dict.get("iter")
        if not isinstance(source_iteration, int) or isinstance(source_iteration, bool):
            raise RuntimeError("Warm-start checkpoint has no valid integer iteration")
        alignment = (metadata or {}).get("go2x5_alignment", metadata or {})
        curriculum = alignment.get("curriculum", {}) if isinstance(alignment, dict) else {}
        self.warm_start_provenance = {
            "mode": "weights_only_warm_start",
            "source_file": os.path.basename(os.path.abspath(path)),
            "source_sha256": self._checkpoint_sha256(path),
            "source_iteration": source_iteration,
            "source_contract_sha256": alignment.get("control_contract_sha256"),
            "source_curriculum_profile": curriculum.get("profile_name"),
            "optimizer_restored": False,
            "history_optimizer_restored": False,
            "exploration_std_restored": False,
            "runner_state_restored": False,
            "environment_state_restored": False,
            "new_run_start_iteration": 0,
            "preserved_current_parameters": preserved_parameters,
            "compatibility": compatibility,
        }
        print(
            "Warm-started compatible model weights from "
            f"{path} (source iteration {source_iteration}); optimizer/std/curriculum reset"
        )
        return dict(self.warm_start_provenance)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        require_full_training_state = bool(
            getattr(getattr(self.env.cfg, 'env', None), 'require_training_metadata', False)
        )
        if hasattr(self.env, "load_training_metadata"):
            self.env.load_training_metadata(loaded_dict.get('metadata'))
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            hist_optimizer_state = loaded_dict.get('hist_encoder_optimizer_state_dict')
            if require_full_training_state and hist_optimizer_state is None:
                raise RuntimeError("Go2-X5 training checkpoint has no history encoder optimizer state")
            if hist_optimizer_state is not None:
                self.alg.hist_encoder_optimizer.load_state_dict(hist_optimizer_state)
        self.current_learning_iteration = loaded_dict['iter']
        if require_full_training_state and 'algorithm_counter' not in loaded_dict:
            raise RuntimeError("Go2-X5 training checkpoint has no PPO algorithm counter")
        if require_full_training_state and 'runner_state' not in loaded_dict:
            raise RuntimeError("Go2-X5 training checkpoint has no runner state")
        self.alg.counter = int(loaded_dict.get('algorithm_counter', loaded_dict['iter']))
        runner_state = loaded_dict.get('runner_state') or {}
        self.tot_timesteps = int(runner_state.get('tot_timesteps', 0))
        self.tot_time = float(runner_state.get('tot_time', 0.0))
        metadata = loaded_dict.get('metadata') or {}
        self.warm_start_provenance = metadata.get('warm_start')
        return loaded_dict['infos']

    def get_inference_policy(self, device=None, stochastic=False):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)

        if not stochastic:
            return self.alg.actor_critic.act_inference
        else:
            return self.alg.actor_critic.act
