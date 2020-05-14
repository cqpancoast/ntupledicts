import tensorflow as tf  #TODO specify imported stuff
from .. import operations as ndops


class TrackPropertiesDataset:
    """A track property dict stored as a tensorflow dataset. Accessible
    from this are the data array, the label array, and the indexing
    track properties for both. Currently assumes a scalar label."""

    def __init__(self, track_prop_dict, label_property, 
            active_data_properties):
        """Initializes using preprocessed tensorflow arrays to set
        fields directly."""

        #TODO decide on field access permissions

        # Determine whether the given track properties dict is valid
        ndops.track_prop_dict_length(track_prop_dict)

        self.track_prop_dict = track_prop_dict
        self.set_label_property(label_property)
        self.set_active_data_properties(active_data_properties)

    def __add__(self, other: TrackPropertiesDataset):
        """Add this TrackPropertiesDataset together with another that
        has the same available data properties, label property, and
        active data properties. If any of these things are not true,
        raises a ValueError."""

        if self.label_property != other.label_property:
            raise ValueError("Chosen label properties do not match.")
        if self.active_data_properties != other.active_data_properties:
            raise ValueError("Active data properties do not match.")

        return TrackPropertiesDataset(
                ndops.add_track_prop_dicts(
                    [self.track_prop_dict, other.track_prop_dict]),
                self.label_property,
                other.active_data_properties)

    def __eq__(self, other: TrackPropertiesDataset):
        """Determines whether two TrackPropertiesDatasets have the same
        active properties, label properties, and available data."""

        return self.label_property == other.label_property and\
                self.active_data_properties == other.active_data_properties and\
                self.track_prop_dict == other.track_prop_dict

    def __ne__ (self, other: TrackPropertiesDataset):
        """Returns whether two TrackPropertiesDatasets are unequal."""

        return not self == other

    def get_data_dim(self, just_active_data=True):
        """Returns the dimension of each element in the data portion of
        the dataset. If active is True, return the number of "active"
        degrees of freedom."""

        return len(self.active_data_properties) if just_active_data\
                else len(self.track_prop_dict.keys())

    def get_num_data(self):
        """Returns the number of elements in this dataset."""

        return len(next(iter(self.track_prop_dict.values())))

    def get_data(self, just_active_data=True):
        """Returns the active portion of the data in this dataset. If
        just_active_data is False, return all data in this dataset as
        a tensorflow array."""

        return self.get_data(self.active_data_properties\
                if just_active_data\
                else list(self.track_properties_dict.keys()))

    def get_data(self, data_properties):
        """Returns data corresponding to the given data properties."""

        return None

    def set_active_data_properties(self, track_properties):
        """Sets the given track properties as the active data properties
        within this dataset. Raises an error if this dataset does not
        have a data property corresponding to each element of the given
        list."""

        for track_property in track_properties:
            if track_property not in self.track_prop_dict.keys()
                raise ValueError("Provided track property " + track_property +
                        "not found in this dataset.")

        self.active_data_properties = track_properties

    def get_labels(self):
        """Returns data corresponding to the given label property."""

        return None

    def set_label_property(self, track_property):
        """Sets the given track property as the label property for this
        dataset. Raises an exception if this dataset does not contain
        the given track property."""

        if track_property not in self.track_prop_dict.keys()
            raise ValueError("Provided track property " + track_property +
                    "not found in this dataset.")

        self.label_property = track_property

    def split(self, split_dist):
        """Splits this dataset into as many separate datasets as given
        by relative sizes in split_dist. Whatever. TODO docs."""

        return [self]


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


#FIXME deprecated
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

