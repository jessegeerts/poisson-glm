from scipy.stats import multivariate_normal, vonmises
import numpy as np


def sim_poisson_spikes(rate, sampling_freq):
    return np.random.poisson(rate / sampling_freq) > 0


def simulate_place_cell_spikes(means, position, max_rate=15, variance=36, sampling_frequency=500):
    firing_rate = simulate_placecell_firing_rate(means, position, max_rate, variance)
    return sim_poisson_spikes(firing_rate, sampling_frequency)


def simulate_hd_cell_spikes(head_dir, pref_dir, max_rate=15, concentration=4, sampling_frequency=500):
    firing_rate = simulate_hdcell_firing_rate(head_dir, pref_dir, max_rate=max_rate, concentration=concentration)
    return sim_poisson_spikes(firing_rate, sampling_frequency)


def simulate_placecell_firing_rate(means, position, max_rate=15, variance=10):
    """Simulate place cell firing rate with a rescaled normal distribution. Works both for 1D or 2D.

    Args:
        means:
        position:
        max_rate:
        variance:

    Returns:

    """

    firing_rate = multivariate_normal(means, variance).pdf(position)
    firing_rate /= firing_rate.max()
    firing_rate *= max_rate
    return firing_rate


def simulate_hdcell_firing_rate(head_dir, pref_dir, max_rate=15, concentration=4):
    """Simulate firing rate of a head direction cell as a Von Mises.

    Args:
        head_dir: Input heading direction.
        pref_dir: Preferred tuning direction.
        max_rate:
        concentration: How narrow is the tuning.

    Returns:

    """
    firing_rate = vonmises(loc=pref_dir, kappa=concentration).pdf(head_dir)
    firing_rate /= firing_rate.max()
    firing_rate *= max_rate
    return firing_rate

