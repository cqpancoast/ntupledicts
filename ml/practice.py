import os
import tensorflow as tf
import pickle
from data/get_data.py import get_data
from build_neuralnet.py import get_model

## In which I practice making a neural network and using data.

# Some constants
hidden_layer_sizes = [14, 6]
num_hidden_layers = len(hidden_layer_sizes)
output_layer_size = 2

event_sets = [os.path("TTbar_whatever")]
track_features = ["trk_pt", "trk_eta", "trk_z0", "trk_chi2rphi", "trk_chi2rz", "trk_fake"]
data_dir = "practicerun"
# data_mix_seed left default in get_data
# save_dist also left default in get_data

model_name = "james"

# Acquiring data
# Data is expected to be stored in three pickled files - train, eval, and test -
# within the chosen data_dir. Their format should be a list of track feature
# vectors.
if !os.path.isdir("data/" + data_dir):
    get_data(event_sets, track_features, save_dir=data_dir)

# Build a model using the data, or pick up the weights
if !os.path.isfile(model_name):
    all_layer_sizes = [len(track_features)].extend(hidden_layer_sizes).append(output_layer_size)
    linear_model = build_model(model_name, all_layer_sizes, data_dir)
else:
    linear_model = load_weights(model_name)

