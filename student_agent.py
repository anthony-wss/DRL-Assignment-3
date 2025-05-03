import gym
import torch
import torch.nn as nn
import torchrl.modules as modules


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            modules.NoisyLinear(7 * 7 * 64, 512),
            nn.ReLU(),
            modules.NoisyLinear(512, 1)
        )
        self.advantage_stream = nn.Sequential(
            modules.NoisyLinear(7 * 7 * 64, 512),
            nn.ReLU(),
            modules.NoisyLinear(512, 12)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        model_state_dict = torch.load("dqn_best.pth", map_location=torch.device('cpu'))
        self.dqn = QNetwork()
        self.dqn.load_state_dict(model_state_dict)
        self.dqn.to(self.device)

    def act(self, observation):
        with torch.no_grad():
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
            action = self.dqn(observation).argmax().item()
        return action
