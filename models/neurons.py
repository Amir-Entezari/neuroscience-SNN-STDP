from pymonntorch import *


class LIF(Behavior):
    def initialize(self, ng):
        """
        Initialize the neuron
        :param ng: neuron group
        :return: None
        """
        # initial parameters in LIF model
        self.R = self.parameter("R", None, required=True)
        self.tau = self.parameter("tau", None, required=True)
        self.u_rest = self.parameter("u_rest", None, required=True)
        self.u_reset = self.parameter("u_reset", None, required=True)
        self.u_init = self.parameter("u_init", f"normal({self.u_reset},0)", required=True)
        self.threshold = self.parameter("threshold", None, required=True)
        self.refractory_T = self.parameter("refractory_T", 0) / ng.network.dt

        # Set parameters
        # self.R = ng.vector(mode=self.R)
        # self.tau = ng.vector(mode=self.tau)
        ng.threshold = ng.vector(mode=self.threshold)

        # initial value of u in neurons
        ng.u_reset = ng.vector(self.u_reset)
        ng.u = ng.vector(mode=self.u_init)
        ng.spike = ng.u > ng.threshold
        ng.u[ng.spike] = self.u_reset

        if not hasattr(ng, 'last_spike'):
            ng.last_spike = ng.vector(-self.refractory_T - 1)

    def forward(self, ng):
        """
        Apply LIF dynamic to neuron groups
        :param ng: neuron group
        :return: None
        """
        # Neuron dynamic
        inp_u = self.R * ng.I * (ng.last_spike < ng.network.iteration - self.refractory_T).byte()
        leakage = ng.u - self.u_rest
        ng.u += ((-leakage + inp_u) / self.tau) * ng.network.dt
        # Firing
        ng.spike = ng.u > ng.threshold
        # Reset
        ng.u[ng.spike] = self.u_reset
        # Save last spike
        ng.last_spike[ng.spike] = ng.network.iteration

