import numpy as np
import torch

from pymonntorch import Behavior


class STDP(Behavior):
    def initialize(self, sg):
        # Trace parameters
        self.tau_pre = self.parameter("tau_pre", None, required=True)  # Presynaptic trace decay constant
        self.tau_post = self.parameter("tau_post", None, required=True)  # Postsynaptic trace decay constant
        # Parameters of A- and A+
        self.eta = self.parameter("eta", 1.0)  # Adding eta for weight change control
        self.w_max = self.parameter("w_max", 1.0)  # Maximum weight for hard bounds
        self.w_min = self.parameter("w_min", -1.0)
        self.learning_rate = self.parameter("learning_rate", None, required=True)
        self.normalization = self.parameter("normalization", True)
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
        # print(sg.x.max(), sg.y.max())
        sg.dW = self.learning_rate * (
                -self.soft_bound_A_minus(sg.W) * sg.src.spike.byte().to(torch.float).reshape(-1, 1).mm(
            sg.y.reshape(1, -1)) \
                + self.soft_bound_A_plus(sg.W) * sg.x.reshape(-1, 1).mm(
            sg.dst.spike.byte().to(torch.float).reshape(1, -1)))

        sg.dW *= abs((self.w_max - sg.W) * (-1 - sg.W))
        if self.normalization:
            sg.dW -= sg.dW.sum(axis=0) / sg.src.size
        sg.dW = sg.dW / (abs(sg.dW).max() or 1)
        sg.W += sg.dW * abs((self.w_max - sg.W) * (self.w_min - sg.W)) / (self.w_max * self.w_min)

        # Reset parameters
        if ((sg.network.iteration - 1) % (sg.network.duration + sg.network.sleep)) == 0:
            sg.src.u = sg.src.u_reset
            sg.dst.u = sg.dst.u_reset
            sg.x = sg.src.vector(0.0)
            sg.y = sg.dst.vector(0.0)

    def soft_bound_A_plus(self, w):
        """ Calculate A+ for soft bounds for a matrix of weights """
        return w * (self.w_max - w) ** self.eta

    def soft_bound_A_minus(self, w):
        """ Calculate A- for soft bounds for a matrix of weights """
        return np.abs(w) ** self.eta

    def hard_bound_A_plus(self, w):
        """ Calculate A+ for hard bounds for a matrix of weights """
        return np.heaviside(self.w_max - w, 0) * (self.w_max - w) ** self.eta

    def hard_bound_A_minus(self, w):
        """ Calculate A- for hard bounds for a matrix of weights """
        return np.heaviside(w, 0) * w ** self.eta


class RSTDP(Behavior):
    def initialize(self, sg):
        # Trace parameters
        self.tau_pre = self.parameter("tau_pre", None, required=True)  # Presynaptic trace decay constant
        self.tau_post = self.parameter("tau_post", None, required=True)  # Postsynaptic trace decay constant
        # Parameters of A- and A+
        self.eta = self.parameter("eta", 1.0)  # Adding eta for weight change control
        self.w_max = self.parameter("w_max", 1.0)  # Maximum weight for hard bounds
        # Learning parameters
        self.learning_rate = self.parameter("learning_rate", None, required=True)
        self.positive_dopamine = self.parameter("positive_dopamine", None, required=True)
        self.negative_dopamine = self.parameter("negative_dopamine", None, required=True)

        # initial value of x and y
        if not hasattr(sg, 'x'):
            sg.x = sg.src.vector(0.0)  # Presynaptic trace
        if not hasattr(sg, 'y'):
            sg.y = sg.dst.vector(0.0)  # Postsynaptic trace

        sg.C = sg.matrix(mode=0.0)
        self.spike_counter = sg.dst.vector()
        self.dopamine_list = sg.dst.vector()

    def forward(self, sg):
        self.spike_counter += sg.dst.spike.byte()

        # update traces:
        sg.x += (-sg.x / self.tau_pre + sg.src.spike.byte()) * sg.network.dt
        sg.y += (-sg.y / self.tau_post + sg.dst.spike.byte()) * sg.network.dt

        # Update weights
        dC = self.learning_rate * (-1 * sg.src.spike.byte().to(torch.float).reshape(-1, 1).mm(sg.y.reshape(1, -1)) \
                                   + 1 * sg.x.reshape(-1, 1).mm(sg.dst.spike.byte().to(torch.float).reshape(1, -1)))
        # dC -= dC.sum(axis=0) / sg.src.size

        sg.C += dC

        if ((sg.network.iteration - 1) % (sg.network.duration + sg.network.sleep)) == 0:
            winners = self.spike_counter.max() == self.spike_counter
            self.dopamine_list = sg.dst.vector(self.negative_dopamine)
            self.dopamine_list[sg.network.curr_data_idx] = self.positive_dopamine
            self.dopamine_list = self.dopamine_list * winners.byte()

            sg.C = sg.C @ torch.diag(self.dopamine_list)
            # sg.C = sg.C/(abs(sg.C).max() or 1)
            sg.C *= abs((self.w_max - sg.W) * (-1 - sg.W))
            sg.C -= sg.C.sum(axis=0) / sg.src.size

            sg.W += sg.C

            # reset parameters
            sg.C = sg.matrix(mode=0.0)
            self.spike_counter = sg.dst.vector()
            self.dopamine_list = sg.dst.vector()
            sg.src.u = sg.src.u_reset
            sg.dst.u = sg.dst.u_reset
            sg.x = sg.src.vector(0.0)
            sg.y = sg.dst.vector(0.0)

    def soft_bound_A_plus(self, w):
        """ Calculate A+ for soft bounds for a matrix of weights """
        return w * (self.w_max - w) ** self.eta

    def soft_bound_A_minus(self, w):
        """ Calculate A- for soft bounds for a matrix of weights """
        return np.abs(w) ** self.eta

    def hard_bound_A_plus(self, w):
        """ Calculate A+ for hard bounds for a matrix of weights """
        return np.heaviside(self.w_max - w, 0) * (self.w_max - w) ** self.eta

    def hard_bound_A_minus(self, w):
        """ Calculate A- for hard bounds for a matrix of weights """
        return np.heaviside(w, 0) * w ** self.eta
