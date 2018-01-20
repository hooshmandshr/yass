import os.path

import numpy as np

from .deconvolute import Deconvolution
from .. import read_config
from ..batch import RecordingsReader


# TODO: comment code, it's not clear what it does
def run(spike_train_clear, templates, spike_index_collision,
        recordings_filename='standarized.bin'):
    """Deconvolute spikes

    Parameters
    ----------
    spike_train_clear: numpy.ndarray (n_clear_spikes, 2)
        A 2D array for clear spikes whose first column indicates the spike
        time and the second column the neuron id determined by the clustering
        algorithm

    templates: numpy.ndarray (n_channels, waveform_size, n_templates)
        A 3D array with the templates

    spike_index_collision: numpy.ndarray (n_collided_spikes, 2)
        A 2D array for collided spikes whose first column indicates the spike
        time and the second column the neuron id determined by the clustering
        algorithm

    recordings_filename: str, optional
        Filename for the recordings which will be used for deconvolution,
        file must be located at CONFIG.data.root_folder/tmp/

    Returns
    -------
    spike_train: numpy.ndarray (n_clear_spikes, 2)
        A 2D array with the spike train, first column indicates the spike
        time and the second column the neuron ID

    Examples
    --------

    .. literalinclude:: ../examples/deconvolute.py
    """
    CONFIG = read_config()

    recordings = RecordingsReader(os.path.join(CONFIG.data.root_folder, 'tmp',
                                               recordings_filename))

    deconv = Deconvolution(CONFIG, np.transpose(templates, [1, 0, 2]),
                           spike_index_collision, recordings)
    spike_train_deconv = deconv.fullMPMU()

    spike_train = np.concatenate((spike_train_deconv, spike_train_clear))

    idx_sort = np.argsort(spike_train[:, 0])
    spike_train = spike_train[idx_sort]

    idx_keep = np.zeros(spike_train.shape[0], 'bool')

    for k in range(templates.shape[2]):
        idx_c = np.where(spike_train[:, 1] == k)[0]
        idx_keep[idx_c[np.concatenate(([True],
                                       np.diff(spike_train[idx_c, 0])
                                       > 1))]] = 1

    spike_train = spike_train[idx_keep]

    return spike_train
