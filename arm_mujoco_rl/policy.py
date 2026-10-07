"""策略网络：checkpoint → 确定性动作。

不需要 isaacgym、不需要 rsl_rl，只要有 torch 就能跑 —— 这是上真机的前提。

结构来自 `arm_reach_config.py:237-246` 的 `actor_hidden_dims = [512,256,128]` +
`activation = 'elu'`，rsl_rl 把它建成 `nn.Sequential(Linear, Act, ..., Linear)`。

checkpoint 里没有输入归一化层，也没有任何隐藏运行时状态（std 只有训练时才用得上），
所以「重建 actor + 载入 actor.* 权重」就完整了。

[可移植] 上真机时原样带走。
"""

import numpy as np
import torch
import torch.nn as nn

import config as C


def _activation():
    if C.ACTIVATION == "elu":
        return nn.ELU()
    if C.ACTIVATION == "relu":
        return nn.ReLU()
    if C.ACTIVATION == "tanh":
        return nn.Tanh()
    raise ValueError("未支持的激活函数: %s" % C.ACTIVATION)


def build_actor():
    """按 config 里的结构建一个空 actor，权重形状必须和 checkpoint 逐层对上。"""
    layers = []
    prev = C.NUM_OBS
    for h in C.ACTOR_HIDDEN_DIMS:
        layers += [nn.Linear(prev, h), _activation()]
        prev = h
    layers.append(nn.Linear(prev, 6))
    return nn.Sequential(*layers)


class Policy:
    """确定性策略。`act()` 返回网络的原始输出（未裁剪、无噪声）。"""

    def __init__(self, actor):
        self.actor = actor

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        # checkpoint 里的键是 "actor.0.weight" 这种（rsl_rl 的 ActorCritic 前缀），
        # 本文件的 actor 是裸 Sequential，键是 "0.weight"，所以要去掉前缀。
        actor_sd = {k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")}
        if not actor_sd:
            raise KeyError("checkpoint 里找不到 actor.* 权重，实际键: %s"
                           % list(sd.keys())[:8])

        actor = build_actor()
        missing, unexpected = actor.load_state_dict(actor_sd, strict=False)
        if missing or unexpected:
            raise RuntimeError("权重对不上 —— missing=%s unexpected=%s"
                               % (list(missing), list(unexpected)))
        actor.to(device).eval()
        return cls(actor)

    def act(self, obs):
        """(30,) 或 (N,30) → (6,) 或 (N,6)。无梯度、不加噪声。"""
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs), dtype=torch.float32,
                                device=next(self.actor.parameters()).device)
            single = x.dim() == 1
            if single:
                x = x.unsqueeze(0)
            y = self.actor(x)
            y = y.squeeze(0) if single else y
            return y.cpu().numpy()
