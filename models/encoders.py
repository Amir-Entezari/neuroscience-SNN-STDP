import matplotlib.pyplot as plt
import torch
from pymonntorch import *
from scipy.stats import norm


class Encoder:
    def __init__(self, dataset, duration):
        self.dataset = dataset
        self.duration = duration
        # self.sleep = sleep
        self.encoded_dataset = None

        if not isinstance(self.dataset, torch.Tensor):
            self.dataset = torch.tensor(self.dataset)

    def __getitem__(self, index):
        return self.encoded_dataset[index]

    def __setitem__(self, index, value):
        self.encoded_dataset[index] = value

    def data_encoder(self, **kwargs):
        pass


class TimeToFirstSpikeEncoder(Encoder):
    def __init__(self, theta, epsilon=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.theta = theta
        self.epsilon = epsilon

        self.encoded_dataset = [self.data_encoder(data) for data in self.dataset]

    def data_encoder(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        if data.dim() > 1:
            print("Data must be converted to vector first.")
        # self.encoded_spikes = torch.zeros((self.duration,) + self.data.shape, dtype=torch.bool)
        encoded_spikes = torch.zeros((self.duration,) + data.shape, dtype=torch.bool)

        data = (data - data.min()) / (data.max() - data.min())
        data = (data * (1 - self.epsilon)) + self.epsilon
        tau = -self.duration / np.log(self.epsilon / self.theta)
        for t in range(self.duration):
            # threshold = self.theta * np.exp(-(t + 1) / tau)
            threshold = np.exp(-(t + 1) / tau)
            encoded_spikes[t, :] = data >= threshold
            data[data >= threshold] = 0

        return encoded_spikes


class NumberEncoder(Encoder):
    def __init__(self, lower_bound=0, upper_bound=10, epsilon=1e-2, std=1, **kwargs):
        super().__init__(**kwargs)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.epsilon = epsilon
        self.std = std

        self.x_values = torch.tensor([np.linspace(self.lower_bound, self.upper_bound, 1000) for mean in
                                 range(self.lower_bound, self.upper_bound + 1)])
        self.normal_dists = torch.tensor([norm.pdf(self.x_values[mean], loc=mean, scale=self.std) for mean in
                                     range(self.lower_bound, self.upper_bound + 1)])

        self.encoded_dataset = [self.data_encoder(data) for data in self.dataset]

    def data_encoder(self, data):

        encoded_spikes = torch.zeros((self.duration, self.upper_bound - self.lower_bound + 1), dtype=torch.bool)

        spike_times = []
        for neuron_id in range(self.lower_bound, self.upper_bound + 1):
            pdf_at_x = self.get_pdf_at_x(data, mean=neuron_id)
            if pdf_at_x > self.epsilon:
                spike_time = ((pdf_at_x - self.normal_dists[0].min()) / (
                        self.normal_dists[0].max() - self.normal_dists[0].min())) * self.duration
                spike_times.append((neuron_id, int(spike_time)))

        for neuron_id, spike_time in spike_times:
            # print(self.duration - spike_time - 1, neuron_id)
            encoded_spikes[self.duration - spike_time - 1, neuron_id - self.lower_bound] = True
        return encoded_spikes

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
        for data in self.dataset:
            for i in range(self.lower_bound, self.upper_bound + 1):
                plt.plot(self.x_values[i], self.normal_dists[i])
                pdf_at_x = self.get_pdf_at_x(data, mean=i)
                pdf_at_x_list.append(pdf_at_x)
                ax.scatter(data, pdf_at_x, color='r')
            ax.axvline(x=data, color='r', linestyle='--', label=f'x={data:.2f}')
        ax.legend()
        ax.set_xlabel('number(neuron_id)')
        ax.set_ylabel('pdf at t (spike priority)')


class PoissonEncoder(Encoder):
    def __init__(self, epsilon=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

        self.encoded_dataset = [self.data_encoder(data) for data in self.dataset]

    def data_encoder(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        if data.dim() > 1:
            print("Data must be converted to vector first.")

        # self.spikes = torch.zeros((self.duration,) + self.data.shape, dtype=torch.bool)
        encoded_spikes = torch.zeros((data.shape[0], self.duration), dtype=torch.bool)

        data = (data - data.min()) / (data.max() - data.min())
        data = (data * (1 - self.epsilon)) + self.epsilon

        for i in range(data.shape[0]):
            spike_times = np.random.poisson(data[i], self.duration)
            for j, t in enumerate(spike_times):
                if t > 0:
                    encoded_spikes[i, j: t + j] = 1
        encoded_spikes = encoded_spikes.T
        return encoded_spikes


class FeedDataset(Behavior):
    def initialize(self, ng):
        self.encoded_dataset = self.parameter("encoded_dataset", None, required=True)
        self.sleep = self.parameter("sleep", None, required=True)

        ng.network.duration = self.encoded_dataset.duration
        ng.network.sleep = self.sleep

    def forward(self, ng):
        # TODO: rewrite the encoded_dataset to ignore multiple dots
        ng.network.curr_data_idx = (ng.network.iteration // (self.encoded_dataset.duration + self.sleep)) % \
                        self.encoded_dataset.dataset.shape[0]

        is_sleep = (ng.network.iteration - 1) % (
                    self.encoded_dataset.duration + self.sleep) < self.encoded_dataset.duration
        ng.spike = is_sleep * self.encoded_dataset[ng.network.curr_data_idx][
            (ng.network.iteration - 1) % self.encoded_dataset.duration]


