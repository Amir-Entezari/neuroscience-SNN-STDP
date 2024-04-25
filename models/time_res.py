from pymonntorch import *


class TimeResolution(Behavior):
    def initialize(self, network):
        network.dt = self.parameter("dt", 1.0)
        network.passed = 0

    def forward(self, network):
        network.passed += network.dt
