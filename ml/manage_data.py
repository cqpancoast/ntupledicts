import tensorflow as tf


def make_datasets_from_track_prop_dict(track_prop_dict, 
        label_property="genuine", split_dist=[.7, .2, .1],
        name="dataset", outputdir="saveddata", seed=None):
    """Creates train, eval, and test datasets each composed of a data
    array and a label array.

    Args:
        track_prop_dict:  a track properties dictionary. Is not altered by
            this function
        label_property:  the field of the track properties dictionary that
            a model will predict. This will typically be "genuine" or "fake"
        split_dist:  tracks will be organized into datasets with these
            relative sizes
        name:  the name to associate with these datasets so they can be
            found and used again
        outputdir:  where to put these datasets
        seed:  a seed to use in array shuffling for reproducability

    Returns:
        Four two-tuples, each containing the data and label components
        of the property names, the training set, the eval set, and the test
        set, all stored as tensorflow tensors
    """

    data_properties = list(track_prop_dict.keys()).remove(label_property)
    label_array = tf.constant(track_prop_dict.pop(label_property)) #TODO labels of more than one element?
    data_array = tf.transpose(tf.constant(list(track_prop_dict.values())))

    # Shuffle the arrays in a reproducable manner
    tf.random.set_seed(seed)
    tf.random.shuffle(data_array)
    tf.random.shuffle(label_array)

    def get_dataset_split_sizes(split_dist, num_tracks):
        """Returns the sizes of the data."""

        split_dist_total = sum(split_dist)
        dataset_split_sizes = list(map(lambda split_val:
            int(split_val * num_tracks / split_dist_total),
            split_dist))

        # Ensure the split sizes add up to the total number of tracks
        dataset_split_sizes[-1] += num_tracks - sum(dataset_split_sizes)

        return dataset_split_sizes

    dataset_split_sizes = get_dataset_split_sizes(split_dist,
            int(tf.shape(data_array)[0]))
    data_datasets = tf.split(data_array, dataset_split_sizes)
    label_datasets = tf.split(label_array, dataset_split_sizes)

    dataset_tuples = list(map(lambda data_dataset, label_dataset:
        (data_dataset, label_dataset),
        data_datasets, label_datasets))

    return [(data_properties, label_property)] + dataset_tuples


def make_track_prop_dict_from_dataset(data, labels, 
        data_properties, label_property):
    """Turns tensorflow data back into a track properties dictionary. For
    example, one might do this to recut the data. Note that, if the dataset
    was created with shuffling, this will not restore the order of the
    original track properties dictionary.
    
    Args:
        data:  a tensorflow tensor of data indexed by tracks and then by
            track properties
        labels:  a tensorflow tensor of data labels indexed by tracks and
            then by label component
        data_properties:  the names of the track properties contained in
            the data array. Should be the same dimension as the track
            properties axis in data
        label_property:  the track property that labels the data. Currently,
            only one label is supported

    Returns:
        A track properties dictionary from data_properties + label_property
        to lists of property values contained in data and labels
    """

    tpd_keys = data_properties.append(label_property)
    tpd_vals = tf.unstack(tf.concat([data, labels], axis=0))

    return dict(map(lambda tpd_key, tpd_val_list:
        (tpd_key, list(tf.eval(tpd_val_list))),
        tpd_keys, tpd_vals))

