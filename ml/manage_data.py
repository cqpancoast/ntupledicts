import tensorflow as tf


def make_datasets_from_track_prop_dict(track_prop_dict, 
        label_property="genuine", tet_dist={"train": .7, "eval": .2, "test": .1},
        name="dataset", outputdir="saveddata", seed=None):
    """Creates train, eval, and test datasets each composed of a feature
    array and a label array.

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
        set, all stored as tensorflow tensors
    """

    feature_properties = list(track_prop_dict.keys()).remove(label_property)
    label_array = tf.constant(track_prop_dict.pop(label_property)) #TODO labels of more than one element?
    feature_array = tf.transpose(tf.constant(list(track_prop_dict.values())))

    # Shuffle the arrays in a reproducable manner
    tf.random.set_seed(seed)
    tf.random.shuffle(feature_array)
    tf.random.shuffle(label_array)

    def get_dataset_split_sizes(tet_dist, num_tracks):
        """Returns the sizes of the data."""

        tet_dist_total = sum(tet_dist.values())
        dataset_split_sizes = list(map(lambda split_val:
            int(split_val * num_tracks / tet_dist_total),
            list(tet_dist.values())))

        # Ensure the split sizes add up to the total number of tracks
        dataset_split_sizes[-1] += num_tracks - sum(dataset_split_sizes)

        return dataset_split_sizes

    dataset_split_sizes = get_dataset_split_sizes(tet_dist,
            int(tf.shape(feature_array)[0]))
    train_features, eval_features, test_features = tf.split(
            feature_array, dataset_split_sizes)
    train_labels, eval_labels, test_labels = tf.split(
            label_array, dataset_split_sizes)

    return (feature_properties, label_property), (train_features, train_labels),\
            (eval_features, eval_labels), (test_features, test_labels)

