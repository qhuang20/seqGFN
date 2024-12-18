
"""TB Model""" 

class TBModel(nn.Module):
    def __init__(self, num_hid, uniform_backward=False):
        super().__init__()
        input_size = n_timesteps + max_len * (vocab_size + 1)  # From state_to_tensor output size
        self.uniform_backward = uniform_backward
        self.mlp = nn.Sequential(
            nn.Linear(input_size, num_hid),
            nn.LeakyReLU(),
            nn.Linear(num_hid, max_actions if uniform_backward else 2 * max_actions),  # Only predict P_F if uniform backward
        )
        self.logZ = nn.Parameter(torch.ones(1))  # log Z is just a single number.

    def forward(self, x):
        logits = self.mlp(x)
        if self.uniform_backward:
            P_F = logits
            P_B = torch.zeros_like(P_F)  # Placeholder for uniform policy
        else:
            # Slice the logits into forward and backward policies
            P_F = logits[..., :max_actions]
            P_B = logits[..., max_actions:]

        return P_F, P_B

def trajectory_balance_loss(logZ, log_P_F, log_P_B, reward):
    """Trajectory balance objective converted into mean squared error loss."""
    # Clip log(reward) to -20 to avoid log(0)
    log_reward = torch.log(reward).clamp(min=-20.0)
    loss = (logZ + log_P_F - log_reward - log_P_B).pow(2)
    return loss
  

