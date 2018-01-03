"""
Running operations in parallel
"""
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from yass.batch import BatchProcessor
from yass.batch import RecordingsReader
from yass.preprocess.filter import butterworth


logging.basicConfig(level=logging.INFO)


path_to_neuropixel_data = (os.path.expanduser('~/data/ucl-neuropixel'
                           '/rawDataSample.bin'))
path_to_filtered_data = (os.path.expanduser('~/data/ucl-neuropixel'
                         '/tmp/filtered.bin'))

# create batch processor for the data
bp = BatchProcessor(path_to_neuropixel_data,
                    dtype='int16', n_channels=385, data_format='wide',
                    max_memory='500MB')

# apply a single channel transformation, each batch will be all observations
# from one channel
filtered = bp.single_channel_apply(butterworth, mode='memory',
                                   output_path=path_to_filtered_data,
                                   low_freq=300, high_factor=0.1,
                                   channels=range(100),
                                   order=3, sampling_freq=30000)
filtered = np.vstack(filtered).T

# let's visualize the results
raw = RecordingsReader(path_to_neuropixel_data, dtype='int16',
                       n_channels=385, data_format='wide')

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(raw[:2000, 0])
ax2.plot(filtered[:2000, 0])
plt.show()
