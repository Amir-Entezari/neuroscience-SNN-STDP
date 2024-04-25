import math

import numpy as np
from pymonntorch import *
import torch
from scipy.stats import norm
import matplotlib.pyplot as plt


class TimeToFirstSpikeEncoder(Behavior):
    def initialize(self, ng):
        self.data = self.parameter("data", None, required=True)
        self.duration = self.parameter("duration", None, required=True)
        self.theta = self.parameter("theta", None, required=True)

        if not isinstance(self.data, torch.Tensor):
            self.data = torch.tensor(self.data)
        self.epsilon = 1e-10
        self.spikes = torch.zeros((self.duration,) + self.data.shape, dtype=torch.bool)
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())
        self.data = (self.data * (1 - self.epsilon)) + self.epsilon
        tau = -self.duration / np.log(self.epsilon / self.theta)
        for t in range(self.duration):
            # threshold = self.theta * np.exp(-(t + 1) / tau)
            threshold = np.exp(-(t + 1) / tau)
            self.spikes[t, :] = self.data >= threshold
            self.data[self.data >= threshold] = 0

        # self.ttfs_priority = self.get_ttfs_priority()

    def forward(self, ng):
        # print(self.spikes[ng.network.iteration % self.duration])
        ng.spike = self.spikes[(ng.network.iteration - 1) % self.duration]

    def get_ttfs_priority(self):
        # Create a list of tuples (value, index)
        priority_list = [(value, index) for index, value in enumerate(self.input)]
        # Sort the list based on value in descending order
        priority_list.sort(reverse=True)

        # Extract the indices from the sorted list
        result = torch.tensor([index for _, index in priority_list])
        return result

