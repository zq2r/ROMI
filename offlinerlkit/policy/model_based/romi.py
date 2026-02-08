import numpy as np
import torch
import torch.nn as nn
import gym
import os

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from collections import defaultdict
from operator import itemgetter
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.policy import MOPOPolicy
from offlinerlkit.dynamics import BaseDynamics


class ROMIPolicy(MOPOPolicy):

    def __init__(
        self,
        dynamics: BaseDynamics,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        hyper_net: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        dynamics_adv_optim: torch.optim.Optimizer,
        hyper_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        adv_weight: float = 0,
        adv_train_steps: int = 1000,
        adv_rollout_batch_size: int = 256,
        adv_rollout_length: int = 5,
        include_ent_in_adv: bool = False,
        scaler: StandardScaler = None,
        device="cpu",
        epsilon=0.01,
    ) -> None:
        super().__init__(
            dynamics, 
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )

        self._dynmics_adv_optim = dynamics_adv_optim
        self._adv_weight = adv_weight
        self._adv_train_steps = adv_train_steps
        self._adv_rollout_batch_size = adv_rollout_batch_size
        self._adv_rollout_length = adv_rollout_length
        self._include_ent_in_adv = include_ent_in_adv
        self.scaler = scaler
        self.device = device
        
        self.hyper_net = hyper_net
        self.hyper_optim = hyper_optim
        
        self.baseline = 0
        
        self.epsilon = epsilon
        
        
    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device), strict=False)
        
    def save(self, path):
        torch.save(self.state_dict(), path)

    def pretrain(self, data: Dict, n_epoch, batch_size, lr, logger) -> None:
        self._bc_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        observations = data["observations"]
        actions = data["actions"]
        sample_num = observations.shape[0]
        idxs = np.arange(sample_num)

        logger.log("Pretraining policy")
        self.actor.train()
        for i_epoch in range(n_epoch):
            np.random.shuffle(idxs)
            sum_loss = 0
            for i_batch in range(sample_num // batch_size):
                batch_obs = observations[i_batch * batch_size: (i_batch + 1) * batch_size]
                batch_act = actions[i_batch * batch_size: (i_batch + 1) * batch_size]
                batch_obs = torch.from_numpy(batch_obs).to(self.device)
                batch_act = torch.from_numpy(batch_act).to(self.device)
                dist = self.actor(batch_obs)
                pred_actions, _ = dist.rsample()
                bc_loss = ((pred_actions - batch_act) ** 2).mean()

                self._bc_optim.zero_grad()
                bc_loss.backward()
                self._bc_optim.step()
                sum_loss += bc_loss.cpu().item()
            print(f"Epoch {i_epoch}, mean bc loss {sum_loss/i_batch}")
        torch.save(self.state_dict(), os.path.join(logger.model_dir, "pretrain.pth"))

    def update_dynamics(
        self, 
        real_buffer, 
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        all_loss_info = {
            "adv_dynamics_update/all_loss": 0, 
            "adv_dynamics_update/sl_loss": 0, 
        }
        self.dynamics.model.train()
        steps = 0
        while steps < self._adv_train_steps:
            init_obss = real_buffer.sample(self._adv_rollout_batch_size)["observations"].cpu().numpy()
            observations = init_obss
            for t in range(self._adv_rollout_length):
                actions = super().select_action(observations)
                sl_observations, sl_actions, sl_next_observations, sl_rewards = \
                    itemgetter("observations", "actions", "next_observations", "rewards")(real_buffer.sample(self._adv_rollout_batch_size))
                next_observations, terminals, loss_info = self.dynamics_step_and_forward(sl_observations, sl_actions, sl_next_observations, sl_rewards)
                for _key in loss_info:
                    all_loss_info[_key] += loss_info[_key]
                steps += 1
                observations = next_observations.copy()
                if steps == 1000:
                    break
        self.dynamics.model.eval()
        return {_key: _value/steps for _key, _value in all_loss_info.items()}
    

    def add_3sigma_noise_batch(self, s, epsilon=0.01, n=10):
        batch_size, dim = s.shape
        device = s.device
        s_norm = torch.norm(s, p=2, dim=1, keepdim=True)
        sigma = (epsilon * s_norm) / 3
        noise = torch.randn(batch_size, n, dim, device=device)
        scaled_noise = noise * sigma.unsqueeze(1)
        perturbed_s = s.unsqueeze(1) + scaled_noise
        perturbed_s = torch.cat([perturbed_s, s.unsqueeze(1)], dim=1)
        return perturbed_s
    
    def dynamics_step_and_forward(
        self,
        sl_observations,
        sl_actions, 
        sl_next_observations,
        sl_rewards,
    ):  
        meta_w = self.hyper_net(sl_observations, sl_actions)
        #compute the supervised loss
        sl_input = torch.cat([sl_observations, sl_actions], dim=-1).cpu().numpy()
        sl_target = torch.cat([sl_next_observations-sl_observations, sl_rewards], dim=-1)
        sl_input = self.dynamics.scaler.transform(sl_input)
        sl_mean, sl_logvar = self.dynamics.model(sl_input)
        sl_loss = (torch.pow(sl_mean - sl_target, 2)).sum(dim=0)      
        
        sl_inv_var = torch.exp(-sl_logvar)
        sl_mse_loss_inv = (torch.pow(sl_mean - sl_target, 2) * sl_inv_var).sum(dim=0)
        sl_var_loss = sl_logvar.sum(dim=0)
        sl_loss = sl_mse_loss_inv + sl_var_loss  
        
        trainable_params = [p for p in self.dynamics.model.parameters() if p.requires_grad]
        dt_val_loss_grad = torch.autograd.grad(torch.mean(sl_loss), trainable_params, retain_graph=True, allow_unused=True) #q直接对actor求梯度，也进行展开
        dt_val_loss_grad = torch.cat([grad.reshape(-1) for grad in dt_val_loss_grad if grad is not None], dim=0).detach()

        
        sl_loss = torch.mean(sl_loss * meta_w.detach())
        self._dynmics_adv_optim.zero_grad()
        sl_loss.backward()
        self._dynmics_adv_optim.step()
        
        
        observations = sl_observations.cpu().numpy()
        actions = sl_actions.cpu().numpy()
        next_observations = sl_next_observations.cpu().numpy()

        obs_act = np.concatenate([observations, actions], axis=-1)
        obs_act = self.dynamics.scaler.transform(obs_act)
        diff_mean, logvar = self.dynamics.model(obs_act)
        observations = torch.tensor(observations).to(diff_mean.device)
        diff_obs, diff_reward = torch.split(diff_mean, [diff_mean.shape[-1]-1, 1], dim=-1)
        mean = torch.cat([diff_obs + observations, diff_reward], dim=-1)
        std = torch.sqrt(torch.exp(logvar))
        
        dist = torch.distributions.Normal(mean, std)
        ensemble_sample = dist.rsample()
        assert ensemble_sample.requires_grad, "Gradient is missing in ensemble_sample!"
        ensemble_size, batch_size, _ = ensemble_sample.shape
        
        pred_next_observations = ensemble_sample[..., :-1] 
        pred_next_observations = pred_next_observations.flatten(0, 1) 
        terminals = self.dynamics.terminal_fn(observations.detach().cpu().numpy(), actions, pred_next_observations.detach().cpu().numpy())
        #terminals = np.zeros_like(actions)
        
        pred_next_actions, pred_next_policy_log_prob = self.actforward(pred_next_observations, deterministic=True)
        next_q = torch.minimum(
                self.critic1_old(pred_next_observations, pred_next_actions), 
                self.critic2_old(pred_next_observations, pred_next_actions)
            )
        if self._include_ent_in_adv:
            next_q = next_q - self._alpha * pred_next_policy_log_prob
        
        next_q = next_q.reshape(ensemble_size, batch_size, -1)

        noise_next_observations = self.add_3sigma_noise_batch(sl_next_observations, epsilon=self.epsilon, n=10)
        batch_size, n, _ = noise_next_observations.shape
        noise_next_observations = noise_next_observations.flatten(0, 1)


        noise_next_actions, noise_next_policy_log_prob = self.actforward(noise_next_observations, deterministic=True)
        noise_next_q = torch.minimum(
                self.critic1_old(noise_next_observations, noise_next_actions), 
                self.critic2_old(noise_next_observations, noise_next_actions)
            )
        noise_next_q = noise_next_q.reshape(batch_size, n, -1)
        noise_next_policy_log_prob = noise_next_policy_log_prob.reshape(batch_size, n, -1)
        noise_next_q, index = torch.min(noise_next_q, dim=1)     


        if self._include_ent_in_adv:
            selected_log_prob = torch.gather(
                noise_next_policy_log_prob, 
                dim=1, 
                index=index.squeeze(-1)
            )
            noise_next_q = noise_next_q - self._alpha * selected_log_prob
            
        robust_loss = ((next_q - noise_next_q)**2).sum(dim=0)
        group_size = 4
        
        dt_rb_grad_list = []
        bs = int(batch_size / group_size)
        for k in range(group_size):
            loss_sum = torch.mean(robust_loss[k*bs:(k+1)*bs, :])
            trainable_params = [p for p in self.dynamics.model.parameters() if p.requires_grad]
            dt_rb_grad = torch.autograd.grad(loss_sum, trainable_params, retain_graph=True, allow_unused=True)
            dt_rb_grad_list.append(torch.cat([grad.reshape(-1) for grad in dt_rb_grad if grad is not None], dim=0))
        
        meta_loss = torch.tensor([0.]).to(self.device) 
        r_tensor = torch.zeros([group_size]).to(self.device)
        for k in range(group_size):
            r_tensor[k] = torch.sum(dt_val_loss_grad * dt_rb_grad_list[k]) / 100 
            meta_loss = meta_loss + torch.mean(meta_w[k*bs:(k+1)*bs] * (r_tensor[k] - self.baseline).detach())
        meta_loss = meta_loss / batch_size

        self.hyper_optim.zero_grad()
        meta_loss.backward()
        self.hyper_optim.step()

        if self.baseline == 0:
            self.baseline = torch.mean(r_tensor)
        else:
            self.baseline = self.baseline - 0.1 * (self.baseline - torch.mean(r_tensor))
        
        return next_observations, terminals, {}

    def rollout(
        self,
        init_obss: np.ndarray,
        rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        for _ in range(rollout_length):
            actions = super().select_action(observations)
            next_observations, rewards, terminals, info = self.dynamics.step(observations, actions)
            rollout_transitions["obss"].append(observations)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]
        
        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions, \
            {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean()}

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self.scaler is not None:
            obs = self.scaler.transform(obs)
        return super().select_action(obs, deterministic)