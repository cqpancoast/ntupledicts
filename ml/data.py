import tensorflow as tf  #TODO specify imported stuff
from .. import operations as ndops
from copy import deepcopy


class TrackPropertiesDataset:
    """A track property dict stored as a tensorflow dataset. Accessible
    from this are the data array, the label array, and the indexing
    track properties for both. Currently assumes a scalar label."""

    def __init__(self, track_prop_dict, label_property,
            active_data_properties):
        """Initializes using preprocessed tensorflow arrays to set
        fields directly."""

        # Determine whether the given track properties dict is valid by using
        # the most basic function checking track properties dict validity
        ndops.track_prop_dict_length(track_prop_dict)

        self._track_prop_dict = track_prop_dict
        self.set_label_property(label_property)
        self.set_active_data_properties(active_data_properties)

    def __add__(self, other):
        """Add this TrackPropertiesDataset together with another that
        has the same available data properties, label property, and
        active data properties. If any of these things are not true,
        raises a ValueError."""

        if self._label_property != other.label_property:
            raise ValueError("Chosen label properties do not match.")
        if self._active_data_properties != other.active_data_properties:
            raise ValueError("Active data properties do not match.")

        return TrackPropertiesDataset(
                ndops.add_track_prop_dicts(
                    [self._track_prop_dict, other.track_prop_dict]),
                self._label_property,
                other.active_data_properties)

    def __eq__(self, other):
        """Determines whether two TrackPropertiesDatasets have the same
        active properties, label properties, and available data."""

        return type(self) == type(other) and\
                self.get_label_property() == other.get_label_property() and\
                self.get_active_data_properties()\
                    == other.get_active_data_properties() and\
                self.to_track_prop_dict() == other.to_track_prop_dict()

    def __ne__ (self, other):
        """Returns whether two TrackPropertiesDatasets are unequal."""

        return not self.__eq__(other)

    def to_track_prop_dict(self):
        """Converts this TrackPropertiesDataset to a track properties
        dict by simply returning the track prop dict it has stored."""

        return deepcopy(self._track_prop_dict)

    def get_data_dim(self, just_active_data=True):
        """Returns the dimension of each element in the data portion of
        the dataset. If active is True, return the number of "active"
        degrees of freedom."""

        return len(self._active_data_properties) if just_active_data\
                else len(self._track_prop_dict.keys())

    def get_num_data(self):
        """Returns the number of elements in this dataset."""

        return len(next(iter(self._track_prop_dict.values())))

    def get_data(self, data_properties=None):
        """Returns data corresponding to the given data properties as
        a tensorflow array. By default, returns the active data."""

        if data_properties is None:
            data_properties = self.get_active_data_properties()

        return tf.transpose(tf.constant(list(map(lambda track_property:
            self._track_prop_dict[track_property],
            data_properties))))

    def get_active_data_properties(self):
        """Returns a list of the current active data properties in this
        TrackPropertiesDataset."""

        return self._active_data_properties

    def get_all_available_data_properties(self):
        """Returns a list of all data properties that this dataset has
        available."""

        return list(self._track_prop_dict.keys())

    def set_active_data_properties(self, track_properties):
        """Sets the given track properties as the active data properties
        within this dataset. Raises an error if this dataset does not
        have a data property corresponding to each element of the given
        list."""

        for track_property in track_properties:
            if track_property not in self.get_all_available_data_properties():
                raise ValueError("Provided track property " + track_property +
                        "not available in this dataset.")

        self._active_data_properties = track_properties

    def get_labels(self):
        """Returns data corresponding to the given label property."""

        return tf.constant(self._track_prop_dict[self._label_property])

    def get_label_property(self):
        """Return this dataset's label property."""

        return self._label_property

    def set_label_property(self, track_property):
        """Sets the given track property as the label property for this
        dataset. Raises an exception if this dataset does not contain
        the given track property."""

        if track_property not in self.get_all_available_data_properties():
            raise ValueError("Provided track property " + track_property +
                    "not found in this dataset.")

        self._label_property = track_property

    def split(self, split_list, shuffle_tracks=False):
        """Returns datasets of number and relative sizes of elements as
        specified in split_dist. Retains the same active properties and
        label property. Does not alter the calling dataset."""

        split_tpds = ndops.split_track_prop_dict(self.to_track_prop_dict(),
                split_list, shuffle_tracks=shuffle_tracks)

        return list(map(lambda split_tpd:
            TrackPropertiesDataset(split_tpd,
                self.get_label_property(),
                self.get_active_data_properties()),
            split_tpds))

