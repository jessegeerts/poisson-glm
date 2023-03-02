from scipy.stats import multivariate_normal
import numpy as np


def simulate_poisson_spikes(rate, sampling_freq):
    return np.random.poisson(rate / sampling_freq) > 0


def simulate_place_cell_spikes(means, position, max_rate=15, variance=36, sampling_frequency=500):
    firing_rate = simulate_placecell_firing_rate(means, position, max_rate, variance)
    return simulate_poisson_spikes(firing_rate, sampling_frequency)


def simulate_placecell_firing_rate(means, position, max_rate=15, variance=10):
    firing_rate = multivariate_normal(means, variance).pdf(position)
    firing_rate /= firing_rate.max()
    firing_rate *= max_rate
    return firing_rate
