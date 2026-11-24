# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
import torch.nn as nn # 【新增导入】导入 PyTorch 的 nn 模块

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import Memory
from rsl_rl.utils import resolve_nn_activation


class ActorCriticRecurrent(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        **kwargs,
    ):
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:  # Only override if the new argument is at its default
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        # 这一步会调用父类 ActorCritic 构建默认的没有 Dropout 的 self.actor 和 self.critic
        super().__init__(
            num_actor_obs=rnn_hidden_dim,
            num_critic_obs=rnn_hidden_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        activation = resolve_nn_activation(activation)

        # =====================================================================
        # 【核心修改区】：覆写父类生成的 self.actor，注入 MC Dropout 层
        # =====================================================================
        dropout_prob = 0.02 
        
        actor_layers = []
        # 输入层 (连接 RNN 的输出和 MLP 的第一层)
        actor_layers.append(nn.Linear(rnn_hidden_dim, actor_hidden_dims[0]))
        actor_layers.append(activation)
        actor_layers.append(nn.Dropout(p=dropout_prob)) # 注入 Dropout

        # 隐藏层
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                # 最后一层直接输出动作，通常不在输出后加激活和 Dropout
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
                actor_layers.append(nn.Dropout(p=dropout_prob)) # 注入 Dropout

        # 用带有 Dropout 的新网络替换掉父类的旧网络
        self.actor = nn.Sequential(*actor_layers)
        # =====================================================================

        self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")
        # 打印一下网络结构确认 Dropout 成功注入
        print(f"Modified Actor MLP with Dropout:\n{self.actor}") 

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        input_a = self.memory_a(observations, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        input_a = self.memory_a(observations)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states