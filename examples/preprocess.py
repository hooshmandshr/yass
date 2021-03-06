import logging

import yass
from yass import preprocess

# configure logging module to get useful information
logging.basicConfig(level=logging.DEBUG)

# set yass configuration parameters
yass.set_config('config_sample.yaml')

# run preprocessor
clear_scores, spike_index_clear, spike_index_collision = preprocess.run()
