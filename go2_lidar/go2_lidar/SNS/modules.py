"""First-order smooth neural surrogate and frozen EF-HRL control interface."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class SNSLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.theta_c = nn.Parameter(torch.tensor(0.0))

    def get_normalized_weight(self):
        weight = self.linear.weight
        row_norm = weight.abs().sum(dim=1, keepdim=True)
        return weight * (self.theta_c.exp() / row_norm.clamp_min(1e-12)).clamp(max=1)

    def forward(self, x):
        return F.linear(x, self.get_normalized_weight(), self.linear.bias)


class SNS_MLP(nn.Module):
    def __init__(self, input_dim=60, hidden_dim=256, output_dim=3, num_layers=3, activation=None):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(SNSLinear(i, o) for i, o in zip(dims, dims[1:]))
        self.act = nn.Tanh() if activation is None else activation

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)

    def lipschitz_bound(self):
        return torch.stack([layer.theta_c for layer in self.layers]).sum().exp()

    def get_lipschitz_penalty(self, c_ub=4.0):
        return (self.lipschitz_bound() / c_ub).clamp_min(1.0)


def input_scale():
    # 5 frames of REASAN locomotion units: gravity, body v, body omega, nav command.
    return torch.tensor([1., 1., 1., 2., 2., 2., .25, .25, .25, 2., 2., .25]).repeat(5)


class ExplicitFeedback(nn.Module):
    """A stateless deployable control tick. Inputs are in physical units."""
    def __init__(self, sns):
        super().__init__()
        # Fold the learned row caps into the three frozen affine maps once.
        for name, layer in zip(("first", "second", "output"), sns.layers):
            self.register_buffer(name + "_weight", layer.get_normalized_weight().detach().clone())
            self.register_buffer(name + "_bias", layer.linear.bias.detach().clone())
        self.register_buffer("scale", input_scale().to(self.first_weight))
        self.register_buffer("eta", self.first_weight.new_tensor([0.6, 0.6, 0.3]))

    def forward(self, history: torch.Tensor, hip_offsets: torch.Tensor, ready: torch.Tensor):
        x = history.flatten(1) * self.scale
        h1 = F.linear(x, self.first_weight, self.first_bias).tanh()
        h2 = F.linear(h1, self.second_weight, self.second_bias).tanh()
        error = F.linear(h2, self.output_weight, self.output_bias)

        # Exact vector-Jacobian product of 1/2 ||f||^2 w.r.t. the latest raw command.
        # It also works inside Isaac's inference_mode and TorchScript on the robot.
        d2 = (error @ self.output_weight) * (1.0 - h2.square())
        d1 = (d2 @ self.second_weight) * (1.0 - h1.square())
        grad_cmd = (d1 @ self.first_weight[:, -3:]) * self.scale[-3:]

        frame = history[:, -1]
        tolerance = 0.2 * frame[:, 4:5].abs()  # measured lateral body velocity / (1 m/s)
        r_hip = 0.75 * (hip_offsets.abs() - tolerance).clamp_min(0).sum(dim=1)
        r_att = 0.5 * frame[:, 6:8].abs().sum(dim=1)
        # Evaluate the exponential and final projection in double precision.
        alpha = torch.expm1((r_hip + r_att).double().clamp_max(300.0))
        alpha = torch.where(ready, alpha, 0.0)
        cmd = frame[:, 9:12] - alpha[:, None] * self.eta * grad_cmd
        xy = cmd[:, :2]
        xy = xy * (3.0 / xy.norm(p=2, dim=1, keepdim=True).clamp_min(3.0))
        cmd = torch.cat((xy, cmd[:, 2:3].clamp(-3.0, 3.0)), dim=1)
        return cmd.to(frame), error, alpha


DEFAULT_MODEL = Path(__file__).with_name("ef_hrl.pt")


if __name__ == "__main__":
    from train_sns import main
    main()
