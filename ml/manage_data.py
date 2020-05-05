#import pickle
import numpy as np


def make_datasets_from_track_properties_dict(track_prop_dict, 
        label_property="genuine", tet_dist={"train": .7, "eval": .2, "test": .1},
        name="dataset", outputdir="saveddata", seed=None):
    """Creates train, eval, and test datasets each composed of a feature
    array and a label array; pickles and saves them somewhere as numpy
    arrays.

    Args:
        track_prop_dict:  a track properties dictionary. Is not altered by
            this function
        label_property:  the field of the track properties dictionary that
            a model will predict. This will typically be "genuine" or "fake"
        tet_dist:  stands for "train", "eval", "test". Tracks will be
            organized into these three datasets with given relative sizes
        name:  the name to associate with these datasets so they can be
            found and used again
        outputdir:  where to put these datasets
        seed:  a seed to use in array shuffling for reproducability

    Returns:
        Four two-tuples, each containing the feature and label components
        of the property names, the training set, the eval set, and the test
        set, all stored as numpy arrays
    """

    np.random.seed(seed)

    feature_properties = list(track_prop_dict.keys()).remove(label_property)
    feature_array = np.random.shuffle(np.array(list(track_prop_dict.values())))
    label_array = np.random.shuffle(np.array(track_prop_dict[label_property]))

    def get_dataset_split_nums(tet_dist, num_tracks):
        """Returns the indices down which to split the dataset using np.split."""

        tet_dist_total = sum(tet_dist.values())
        return map(lambda split_val: float(split_val) * num_tracks / tet_dist_total,
                list(tet_dist.values())[:-1])

    dataset_split_nums = get_dataset_split_nums(tet_dist,
            np.ndarray.size(feature_array))
    train_features, eval_features, test_features = np.split(
            feature_array, dataset_split_nums, axis=1)
    train_labels, eval_labels, test_labels = np.split(
            label_array, dataset_split_nums, axis=1)

    return (feature_properties, label_property), (train_features, train_labels),\
            (eval_features, eval_labels), (test_features, test_labels)

