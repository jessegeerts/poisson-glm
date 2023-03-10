from scipy.stats import multivariate_normal, vonmises
import numpy as np


class PoissonCell(object):
    """Base class for simulated Poisson neurons.
    """
    def __init__(self, max_rate=20.):
        self.max_rate = max_rate

    @staticmethod
    def sim_poisson_spikes(rate, sampling_freq):
        return np.random.poisson(rate / sampling_freq) > 0


class PlaceCell(PoissonCell):
    """Class for simulating simple place cells with Gaussian fields.
    """
    def __init__(self, pref_pos, pos_var, max_rate=20.):
        super().__init__(max_rate=max_rate)
        self.pref_pos = pref_pos
        self.pos_var = pos_var

    def get_firing_rate(self, position):
        firing_rate = multivariate_normal(self.pref_pos, self.pos_var).pdf(position)
        firing_rate /= firing_rate.max()
        firing_rate *= self.max_rate
        return firing_rate

    def simulate_spikes(self, position, sampling_freq):
        firing_rate = self.get_firing_rate(position)
        return self.sim_poisson_spikes(firing_rate, sampling_freq)


class HeadDirectionCell(PoissonCell):
    """Class for simulating a head direction cell as a Von Mises.
    """
    def __init__(self, pref_dir, precision, max_rate=20.):
        super().__init__(max_rate)
        self.pref_dir = pref_dir
        self.precision = precision

    def get_firing_rate(self, head_dir):
        firing_rate = vonmises(loc=self.pref_dir, kappa=self.precision).pdf(head_dir)
        firing_rate /= firing_rate.max()
        firing_rate *= self.max_rate
        return firing_rate

    def simulate_spikes(self, head_dir, sampling_freq):
        firing_rate = self.get_firing_rate(head_dir)
        return self.sim_poisson_spikes(firing_rate, sampling_freq)


class DirectionalPlaceCell(PoissonCell):
    def __init__(self, pref_pos, pos_var, pref_dir, dir_precision, max_rate=20):
        super().__init__(max_rate=max_rate)
        self.place_input = PlaceCell(pref_pos, pos_var)
        self.dir_input = HeadDirectionCell(pref_dir, dir_precision)

    def get_firing_rate(self, pos, head_dir):
        firing_rate = self.place_input.get_firing_rate(pos) * self.dir_input.get_firing_rate(head_dir)
        firing_rate /= firing_rate.max()
        firing_rate *= self.max_rate
        return firing_rate


def sim_place_cell_spikes(position, mean=None, max_rate=15, variance=36, sampling_frequency=500):
    """Simulate place cell spikes. If no field location is chosen, choose random location.

    Args:
        position:
        mean:
        max_rate:
        variance:
        sampling_frequency:

    Returns:

    """
    if mean is None:
        if position.ndim == 1:
            mean = np.random.uniform(position.min, position.max)
        elif position.ndim == 2:
            meanx = np.random.uniform(position[0].min, position[0].max)
            meany = np.random.uniform(position[1].min, position[1].max)
            mean = np.array([meanx, meany])

    c = PlaceCell(mean, variance, max_rate=max_rate)
    firing_rate = c.get_firing_rate(position)
    return c.sim_poisson_spikes(firing_rate, sampling_frequency)


def sim_hd_cell_spikes(head_dir, pref_dir=None, max_rate=15., concentration=4., sampling_frequency=500, return_object=False):
    """Simulate head direction cell spikes. if no preferred direction is input, choose random direction.

    Args:
        head_dir:
        pref_dir:
        max_rate:
        concentration:
        sampling_frequency:

    Returns:

    """
    if pref_dir is None:
        pref_dir = np.random.uniform(0, 2 * np.pi)

    c = HeadDirectionCell(pref_dir=pref_dir, precision=concentration, max_rate=max_rate)
    firing_rate = c.get_firing_rate(head_dir)
    spikes = c.sim_poisson_spikes(firing_rate, sampling_frequency)
    if return_object:
        return spikes, c
    return spikes


def sim_dir_placecell_spikes(pos, head_dir, pref_pos=None, pref_dir=None, pos_var=36, dir_precision=4, max_rate=20,
                             sampling_frequency=500):
    if pref_pos is None:
        if pos.ndim == 1:
            pref_pos = np.random.uniform(pos.min, pos.max)
        elif pos.ndim == 2:
            meanx = np.random.uniform(pos[0].min, pos[0].max)
            meany = np.random.uniform(pos[1].min, pos[1].max)
            pref_pos = np.array([meanx, meany])
    if pref_dir is None:
        pref_dir = np.random.uniform(0, 2 * np.pi)

    c = DirectionalPlaceCell(pref_pos, pos_var, pref_dir, dir_precision, max_rate=max_rate)
    firing_rate = c.get_firing_rate(pos, head_dir)
    return c.sim_poisson_spikes(firing_rate, sampling_frequency)

