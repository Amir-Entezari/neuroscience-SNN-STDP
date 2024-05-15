from pymonntorch import Behavior


class ActivityRecorder(Behavior):
    def initialize(self, ng):
        num_spikes = ng.spike.sum()
        ng.activity = num_spikes / ng.size

    def forward(self, ng):
        num_spikes = ng.spike.sum()
        ng.activity = num_spikes / ng.size
