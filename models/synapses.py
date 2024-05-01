import random

import torch
import numpy as np
from pymonntorch import Behavior


class SimpleSynapse(Behavior):
    def initialize(self, sg):
        self.mode = self.parameter("mode", "normal(1.0,0.0)")
        self.alpha = self.parameter("alpha", 1.0)
        sg.W = sg.matrix(mode=self.mode)
        sg.I = sg.dst.vector()

    def forward(self, sg):
        pre_spike = sg.src.spike
        sg.I = torch.sum(sg.W[pre_spike], axis=0) * self.alpha


class FullyConnectedSynapse(Behavior):
    """
    Fully connected synapse class that connect all neurons in a source and destination.
    """

    def initialize(self, sg):
        # Weight parameters
        self.j0 = self.parameter("j0", None, required=True)
        self.variance = self.parameter("variance", None)
        self.alpha = self.parameter("alpha", 1.0)

        # Trace parameters
        self.tau_pre = self.parameter("tau_pre", None, required=True)  # Presynaptic trace decay constant
        self.tau_post = self.parameter("tau_post", None, required=True)  # Postsynaptic trace decay constant
        # Parameters of A- and A+
        self.eta = self.parameter("eta", 1.0)  # Adding eta for weight change control
        self.w_max = self.parameter("w_max", 1.0)  # Maximum weight for hard bounds

        self.N = sg.src.size
        mean = self.j0 / self.N

        if self.variance is None:
            self.variance = self.j0 / np.sqrt(self.N)
        else:
            self.variance = abs(mean) * self.variance

        # initial value of x and y
        if not hasattr(sg, 'x'):
            sg.x = sg.src.vector(0.0)  # Presynaptic trace
        if not hasattr(sg, 'y'):
            sg.y = sg.dst.vector(0.0)  # Postsynaptic trace

        sg.W = sg.matrix(mode=f"normal({mean},{self.variance})")
        # Make the diagonal zero
        if sg.src == sg.dst:
            sg.W.fill_diagonal_(0)
        sg.I = sg.dst.vector()

    def forward(self, sg):
        pre_spike = sg.src.spike
        # sg.I = torch.sum(sg.W[pre_spike], axis=0)
        sg.I += torch.sum(sg.W[pre_spike], axis=0) - sg.I * self.alpha

        # update traces:
        sg.x += (-sg.x / self.tau_pre + sg.src.spike.byte()) * sg.network.dt
        sg.y += (-sg.y / self.tau_post + sg.dst.spike.byte()) * sg.network.dt

        # Update weights
        sg.W += sg.y * -self.soft_bound_A_minus(sg.W) * sg.dst.spike.byte() + sg.x * self.soft_bound_A_plus(sg.W) * sg.src.spike.byte()

    def soft_bound_A_plus(self, w):
        """ Calculate A+ for soft bounds for a matrix of weights """
        return (self.w_max - w) ** self.eta

    def soft_bound_A_minus(self, w):
        """ Calculate A- for soft bounds for a matrix of weights """
        return w ** self.eta

    def hard_bound_A_plus(self, w):
        """ Calculate A+ for hard bounds for a matrix of weights """
        return np.heaviside(self.w_max - w, 0) * (self.w_max - w) ** self.eta

    def hard_bound_A_minus(self, w):
        """ Calculate A- for hard bounds for a matrix of weights """
        return np.heaviside(w, 0) * w ** self.eta


class RandomConnectedFixedProbSynapse(Behavior):
    """
    Random connected with fixed coupling probability synapse class that connect neurons in a source and destination
    with a probability
    """

    def initialize(self, sg):
        # Parameters:
        self.j0 = self.parameter("j0", None, required=True)
        self.variance = self.parameter("variance", None)
        self.p = self.parameter("p", None, required=True)
        self.alpha = self.parameter("alpha", 1.0)

        self.N = sg.src.size
        mean = self.j0 / (self.p * self.N)

        if self.variance is None:
            self.variance = self.p * (1 - self.p) * self.N
        else:
            self.variance = abs(mean) * self.variance
        # variance = self.p * (1 - self.p) * self.N

        sg.W = sg.matrix(mode=f"normal({mean},{self.variance})")
        sg.W[torch.rand_like(sg.W) > self.p] = 0
        # Make the diagonal zero
        if sg.src == sg.dst:
            sg.W.fill_diagonal_(0)

        sg.I = sg.dst.vector()

    def forward(self, sg):
        pre_spike = sg.src.spike
        # sg.I = torch.sum(sg.W[pre_spike], axis=0)
        sg.I += torch.sum(sg.W[pre_spike], axis=0) - sg.I * self.alpha


class RandomConnectedFixedInputSynapse(Behavior):
    """
    Random with fixed number of presynaptic partners connected synapse class that connect fixed number of neurons in
    a source and destination.
    """

    def initialize(self, sg):
        # Parameters:
        self.j0 = self.parameter("j0", None, required=True)
        self.variance = self.parameter("variance", None, required=True)
        self.n = self.parameter("n", None, required=True)
        self.alpha = self.parameter("alpha", 1.0)

        self.N = sg.src.size

        mean = self.j0 / self.n
        sg.W = sg.matrix(mode=f"normal({mean},{abs(mean) * self.variance})")

        # Create a mask of which elements set to zero
        mask = self.create_random_zero_mask(sg.W.shape, self.n)
        # Apply the mask to the matrix
        sg.W[mask] = 0
        # Make the diagonal zero
        if sg.src == sg.dst:
            sg.W.fill_diagonal_(0)

        sg.I = sg.dst.vector()

    def forward(self, sg):
        pre_spike = sg.src.spike
        # sg.I = torch.sum(sg.W[pre_spike], axis=0)
        sg.I += torch.sum(sg.W[pre_spike], axis=0) - sg.I * self.alpha

    def create_random_zero_mask(self, shape, n):
        mask = torch.full(shape, True, dtype=torch.bool)
        rows, cols = shape

        for col in range(cols):
            selected_rows = torch.randperm(rows)[:n]  # Randomly choose n indices for each column
            mask[selected_rows, col] = False  # Set selected elements to 1

        return mask  # Apply the mask to the original matrix
