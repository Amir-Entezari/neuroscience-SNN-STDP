import numpy as np

from pymonntorch import Behavior


class STDP(Behavior):
    def initialize(self, sg):
        # Trace parameters
        self.tau_pre = self.parameter("tau_pre", None, required=True)  # Presynaptic trace decay constant
        self.tau_post = self.parameter("tau_post", None, required=True)  # Postsynaptic trace decay constant
        # Parameters of A- and A+
        self.eta = self.parameter("eta", 1.0)  # Adding eta for weight change control
        self.w_max = self.parameter("w_max", 1.0)  # Maximum weight for hard bounds

        # initial value of x and y
        if not hasattr(sg, 'x'):
            sg.x = sg.src.vector(0.0)  # Presynaptic trace
        if not hasattr(sg, 'y'):
            sg.y = sg.dst.vector(0.0)  # Postsynaptic trace

    def forward(self, sg):
        # update traces:
        sg.x += (-sg.x / self.tau_pre + sg.src.spike.byte()) * sg.network.dt
        sg.y += (-sg.y / self.tau_post + sg.dst.spike.byte()) * sg.network.dt

        # Update weights
        sg.W += sg.y * -self.soft_bound_A_minus(sg.W) * sg.dst.spike.byte() + sg.x * self.soft_bound_A_plus(
            sg.W) * sg.src.spike.byte()

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
