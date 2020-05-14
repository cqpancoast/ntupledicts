import tensorflow as tf


class TrackPropertiesDataset:
    """A track property dict stored as a tensorflow dataset. Accessible
    from this are the data array, the label array, and the indexing
    track properties for both. Currently assumes a scalar label."""

    def __init__(self, data, data_properties, labels, label_property):
        """Initializes using preprocessed tensorflow arrays to set
        fields directly."""

        self.data = data
        self.data_properties = data_properties
        self.labels = labels
        self.label_property = label_property

    def get_data_dim(self):
        """Returns the dimension of each element in the data portion of
        the dataset."""

        return int(tf.shape(self.data)[1])

    def get_num_data(self):
        """Returns the number of elements in this dataset."""

        return int(tf.shape(self.data)[0])


def make_datasets_from_track_prop_dict(track_prop_dict,
        data_properties=None, label_property="genuine",
        split_dist=[.7, .2, .1]):
    """Makes one or more datasets from the given track properties dict,
    given a label property that tells it which track property to use as
    the label and a split distribution that tells it how many datasets
    to make and their relative sizes. NOTE THAT THIS DOES NOT SHUFFLE
    THE DATA FOR YOU.
    
    Args:
        track_prop_dict: a track properties dictionary. Not altered by
            this function
        data_properties: the properties to pull from track_prop_dict to
            put into the dataset. If none, pulls all data into dataset
        label_property: the field of the track properties dictionary
            that a model will predict. This will typically be "genuine"
            or "fake"
        split_dist: tracks will be organized into datasets with relative
            sizes. [.7, .3] and [700, 300] produce identical output

    Returns:
        A list of two-tuples, beginning with the data and label property
        names and then one two-tuple for each dataset created
    """

    if data_properties is None:
        data_properties = list(track_prop_dict.keys())
        data_properties.remove(label_property)
    label_array = tf.constant(track_prop_dict.pop(label_property)) #TODO labels of more than one element?
    data_array = tf.transpose(tf.constant(list(track_prop_dict.values())))

    def get_dataset_split_sizes(split_dist, num_tracks):
        """Returns the sizes of data by normalizing the provided split
        distribution and mutliplying by the number of tracks in such
        a way that the resulting sizes add up to the original tracks."""

        split_dist_total = sum(split_dist)
        dataset_split_sizes = list(map(lambda split_val:
            int(split_val * num_tracks / split_dist_total),
            split_dist))

        # Ensure the split sizes add up to the total number of tracks
        dataset_split_sizes[-1] += num_tracks - sum(dataset_split_sizes)

        return dataset_split_sizes

    dataset_split_sizes = get_dataset_split_sizes(split_dist,
            int(tf.shape(data_array)[0]))
    data_split = tf.split(data_array, dataset_split_sizes)
    label_split = tf.split(label_array, dataset_split_sizes)

    datasets = list(map(lambda data, labels:
        TrackPropertiesDataset(data, data_properties, labels, label_property),
        data_split, label_split))

    return datasets


def make_track_prop_dict_from_dataset(dataset):
    """Turns tensorflow data back into a track properties dictionary.
    For example, one might do this to recut the data. Note that this
    recasting will not preserve the data type (everything is cast to a
    float) of an original reference track properties dictionary.

    Args:
        dataset: a TrackPropertiesDataset

    Returns:
        A track properties dict from data_properties + label_property to
        lists of property values contained in data and labels
    """

    tpd_keys = dataset.data_properties + [dataset.label_property]
    tpd_value_lists = tf.transpose(tf.unstack(tf.concat(
        [dataset.data,
            tf.stack([tf.cast(dataset.labels, tf.float32)], axis=1)],
        axis=1)))

    return dict(map(lambda tpd_key, tpd_val_list:
        (tpd_key, list(tpd_val_list.numpy())),
        tpd_keys, tpd_value_lists))

