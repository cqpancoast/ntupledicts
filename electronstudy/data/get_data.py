from sys import argv
import uproot


def get_data(event_sets, feature_list,
        save_dir="data", track_mix_seed=None, save_dist=[.7, .2, .1]):
    """
    Processes track data from event sets into feature vectors, then distributes
    those vectors into train, eval, and test sets for a machine learning algo.
   
    Args:
        event_sets: a list of paths to ntuples containing events with tracks
        feature_vector: a list of features to be extracted from the tracks
        save_dir: the directory in which to make train, eval, and test dirs
        track_mix_seed: the seed used for the random mixing of the tracks.
            If none, will choose one randomly
        save_dist: a size-three distribution where the first, second, and
            third entries correspond to proportions of data to go to train,
            eval, and test directories
    """


def extract_track_features(ntuple_path, feature_list):
    """
    Takes in the path to an ntuple and a list of features to be extracted from
    it, returns a list of feature vectors; one feature vector for each track.
    """

