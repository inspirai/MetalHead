import torch
import torch.nn as nn
import torch.utils.data
from torch import autograd

from rsl_rl.utils import utils

DISC_LOGIT_INIT_SCALE = 1.0

class AMPDiscriminator(nn.Module):
    def __init__(  # 86, 2.0, [1024, 512], 'cuda:0', 0.3
            self, input_dim, amp_reward_coef, hidden_layer_sizes, device, task_reward_lerp=0.0):
        super(AMPDiscriminator, self).__init__()

        self.device = device  # 'cuda:0'
        self.input_dim = input_dim  # [1024, 512]

        self.amp_reward_coef = amp_reward_coef  # 2.0
        amp_layers = []
        curr_in_dim = input_dim  # 86
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                pass
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)

        torch.nn.init.uniform_(self.amp_linear.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
        torch.nn.init.zeros_(self.amp_linear.bias)

        self.trunk.train()
        self.amp_linear.train()

        self.task_reward_lerp = task_reward_lerp

    def forward(self, x):
        h = self.trunk(x)
        d = torch.tanh(self.amp_linear(h))
        return d

    def compute_grad_pen(self,
                         expert_state,
                         expert_next_state,
                         lambda_=10):
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_amp_reward(
            self, state, next_state, task_reward, normalizer=None):
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)

            d = torch.tanh(self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1))))

            # prob = 1 / (1 + torch.exp(-d))
            # disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            # reward = disc_r * self.amp_reward_coef  # *=2

            reward = self.amp_reward_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)

            if self.task_reward_lerp > 0:
                reward_d, reward_t = self._lerp_reward(reward, task_reward.unsqueeze(-1))
                reward = reward_d + reward_t
            self.train()
        return reward.squeeze(), d, reward_d.squeeze(), reward_t.squeeze()

    def _lerp_reward(self, disc_r, task_r):
        # r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        dr = (1.0 - self.task_reward_lerp) * disc_r
        tr = self.task_reward_lerp * task_r
        return dr, tr

    def get_disc_logit_weights(self):
        return torch.flatten(self.amp_linear.weight)

    def get_disc_weights(self):
        weights = []
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                weights.append(torch.flatten(m.weight))

        weights.append(torch.flatten(self.amp_linear.weight))
        return weights