import matplotlib.pyplot as plt
from pymonntorch import *
from scipy.stats import norm


class TimeToFirstSpikeEncoder(Behavior):
    def initialize(self, ng):
        self.data = self.parameter("data", None, required=True)
        self.duration = self.parameter("duration", None, required=True)
        self.sleep = self.parameter("sleep", None, required=True)
        self.theta = self.parameter("theta", None, required=True)
        self.epsilon = self.parameter("epsilon", 1e-3)

        if not isinstance(self.data, torch.Tensor):
            self.data = torch.tensor(self.data)
        if self.data.dim() > 1:
            print("Data must be converted to vector first.")
        # self.encoded_spikes = torch.zeros((self.duration,) + self.data.shape, dtype=torch.bool)
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
        is_sleep = (ng.network.iteration - 1) % (self.duration + self.sleep) < self.duration
        ng.spike = is_sleep * self.spikes[(ng.network.iteration - 1) % self.duration]

    def get_ttfs_priority(self):
        # Create a list of tuples (value, index)
        priority_list = [(value, index) for index, value in enumerate(self.input)]
        # Sort the list based on value in descending order
        priority_list.sort(reverse=True)

        # Extract the indices from the sorted list
        result = torch.tensor([index for _, index in priority_list])
        return result


class NumberEncoder(Behavior):
    def initialize(self, ng):
        self.num = self.parameter("num", None, required=True)
        self.upper_bound = self.parameter("upper_bound", 10)
        self.lower_bound = self.parameter("lower_bound", 0)
        self.duration = self.parameter("duration", None, required=True)
        self.epsilon = self.parameter("epsilon", 1e-2)
        self.std = self.parameter("std", 1.0)

        # domain = self.upper_bound - self.lower_bound
        # self.x_values = [np.linspace(mean - (domain / 2) * self.std, mean + (domain / 2) * self.std, 1000) for mean in
        #                  range(self.lower_bound, self.upper_bound + 1)]
        self.x_values = torch.tensor([np.linspace(self.lower_bound, self.upper_bound, 1000) for mean in
                                      range(self.lower_bound, self.upper_bound + 1)])
        self.normal_dists = torch.tensor([norm.pdf(self.x_values[mean], loc=mean, scale=self.std) for mean in
                                          range(self.lower_bound, self.upper_bound + 1)])

        self.spikes = torch.zeros((self.duration, self.upper_bound - self.lower_bound + 1), dtype=torch.bool)
        self.scheduler()

    def forward(self, ng):
        ng.spike = self.spikes[(ng.network.iteration - 1) % self.duration]

    def scheduler(self):
        spike_times = []
        for neuron_id in range(self.lower_bound, self.upper_bound + 1):
            pdf_at_x = self.get_pdf_at_x(self.num, mean=neuron_id)
            if pdf_at_x > self.epsilon:
                spike_time = ((pdf_at_x - self.normal_dists[0].min()) / (
                        self.normal_dists[0].max() - self.normal_dists[0].min())) * self.duration
                spike_times.append((neuron_id, int(spike_time)))

        for neuron_id, spike_time in spike_times:
            # print(self.duration - spike_time - 1, neuron_id)
            self.spikes[self.duration - spike_time - 1, neuron_id - self.lower_bound] = True

    def create_normal_dist(self, mean, std=1):
        x_values = np.linspace(mean - 5 * std, mean + 5 * std, 1000)
        # Calculate the probability density at each x value
        pdf_values = norm.pdf(x_values, loc=mean, scale=std)
        return pdf_values

    def get_pdf_at_x(self, num, mean):
        # Calculate the probability density at x
        pdf_at_x = norm.pdf(num, loc=mean, scale=self.std)
        return pdf_at_x

    def plot_x(self, ax):
        pdf_at_x_list = []
        # plt.figure(figsize=(12, 6))
        for i in range(self.lower_bound, self.upper_bound + 1):
            plt.plot(self.x_values[i], self.normal_dists[i])
            pdf_at_x = self.get_pdf_at_x(self.num, mean=i)
            pdf_at_x_list.append(pdf_at_x)
            ax.scatter(self.num, pdf_at_x, color='r', label=f'({self.num}, {pdf_at_x:.2f})')
        ax.set_xlabel('number(neuron_id)')
        ax.set_ylabel('spike t')
        ax.axvline(x=self.num, color='r', linestyle='--', label=f'x={self.num}')


class PoissonEncoder(Behavior):
    def initialize(self, ng):
        self.data = self.parameter("data", None, required=True)
        self.duration = self.parameter("duration", None, required=True)
        self.sleep = self.parameter("sleep", None, required=True)
        self.epsilon = self.parameter("epsilon", 1e-3)

        if not isinstance(self.data, torch.Tensor):
            self.data = torch.tensor(self.data)
        if self.data.dim() > 1:
            print("Data must be converted to vector first.")

        # self.spikes = torch.zeros((self.duration,) + self.data.shape, dtype=torch.bool)
        self.spikes = torch.zeros((self.data.shape[0], self.duration), dtype=torch.bool)

        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())
        self.data = (self.data * (1 - self.epsilon)) + self.epsilon

        # for i in range(self.data.shape[0]):
        #     spike_times = np.random.poisson(self.data[i], self.duration)
        #     for j, t in enumerate(spike_times):
        #         if t > 0:
        #             self.spikes[j: t + j, i] = 1
        for i in range(self.data.shape[0]):
            spike_times = np.random.poisson(self.data[i], self.duration)
            for j, t in enumerate(spike_times):
                if t > 0:
                    self.spikes[i, j: t + j] = 1
        self.spikes = self.spikes.T

    def forward(self, ng):
        is_sleep = (ng.network.iteration - 1) % (self.duration + self.sleep) < self.duration
        ng.spike = is_sleep * self.spikes[(ng.network.iteration - 1) % self.duration]
