from tensorflow import constant as tfconst
from tensorflow import transpose as tftrans
from tensorflow.keras.utils import normalize as tfnorm
from tensorflow import float64
from .. import operations as ndops
from copy import deepcopy


class TrackPropertiesDataset:
    """A track property dict disguised as a tensorflow dataset.

    The properties of both the data and the label are set dynamically
    such that this dataset can store more property data than a model
    will train on. This is done by introducing "active properties",
    which are a subset of the total data stored.

        print(list(tpd.keys()))  # A track properties dict with keys
                                 # ["pt", "chi2", "nstub", "genuine"]
        # set genuine as a label property
        tpds = TrackPropertiesDataset(tpd, "genuine")
        tpds.get_available_data_properties()  # ["pt", "chi2", "nstub",
                                              # "genuine"]
        # by default, the label property is not an active data property
        tpds.get_active_data_properties()  # ["pt", "chi2", "nstub"]
        tpds.set_active_data_properties(["pt"])
        tpds.get_active_data_properties()  # ["pt"]
        tpds.set_active_data_properties(["chi2", "matchtp_pdgid"])
            # ValueError: matchtp_pdgid not an available data property

    By default, self.get_data() returns data normalized for each track
    property for maximum model compatibility, though this can be
    disabled with the kwarg normalize=False.

    This dataset can also store predictions, accept selector dicts to
    preform cuts, and be split into multiple datasets of the same form
    but of different size.
    """

    def __init__(self, track_prop_dict, label_property,
                 active_data_properties=[], prediction_dict={}):
        """Initializes this TrackPropertiesDataset with its track
        properties dict, its access settings, and predictions, if any.

        Args:
            track_prop_dict: a track properties dict.
            label_property: a track property in track_prop_dict.
            active_data_properties: a list that is a subset of the
                properties in track_prop_dict, or None. If None, all
                available properties become active save for the label
                property.
            prediction_dict: a dictionary from prediction names
                (typically the name of the model) to prediction lists.
        """

        # Determine whether the given track properties dict is valid by using
        # the most basic function checking track properties dict validity
        ndops.track_prop_dict_length(track_prop_dict)

        self._track_prop_dict = track_prop_dict
        self.set_label_property(label_property)
        self.set_active_data_properties(active_data_properties)
        self._predictions = prediction_dict

    def __add__(self, other):
        """Add this TrackPropertiesDataset together with another.

        Args:
            other: a TrackPropertiesDataset that has the same available
                data properties, label property, active data properties,
                and prediction names.

        Returns:
            A TrackPropertiesDataset that has the tracks of this
            dataset and the other one, with the same settings.

        Raises:
            ValueError: if the other dataset has different data
                properties, label property, active data properties, or
                prediction names than this one.
        """

        if self.get_label_property() != other.get_label_property():
            raise ValueError("Chosen label properties do not match.")
        if self.get_active_data_properties() \
                != other.get_active_data_properties():
            raise ValueError("Active data properties do not match.")
        if self.get_available_data_properties() \
                != other.get_available_data_properties():
            raise ValueError("Available data properties do not match.")
        if self.get_all_prediction_names() \
                != other.get_all_prediction_names():
            raise ValueError("Available prediction names do not match.")

        return TrackPropertiesDataset(
            ndops.add_track_prop_dicts(
                [self._track_prop_dict, other._track_prop_dict]),
            self._label_property,
            other.active_data_properties,
            ndops.add_track_prop_dicts(
                [self._predictions, other._predictions]))

    def __eq__(self, other):
        """Determines whether two TrackPropertiesDatasets have the same
        active properties, label properties, available data, and
        available predictions."""

        return isinstance(self, type(other)) and \
               self.get_label_property() == other.get_label_property() and \
               self.get_active_data_properties() \
               == other.get_active_data_properties() and \
               self.to_track_prop_dict(include_preds=True) \
               == other.to_track_prop_dict(include_preds=True)

    def __ne__(self, other):
        """Returns whether two TrackPropertiesDatasets are unequal,
        defined by negating __eq__."""

        return not self.__eq__(other)

    def get_data_dim(self, just_active_data=True):
        """Returns the dimension of each element in the data portion of
        the dataset. If active is True, return the number of "active"
        degrees of freedom."""

        return len(self.get_active_data_properties()) if just_active_data \
            else len(self.get_available_data_properties())

    def size(self):
        """Returns the number of tracks in this dataset."""

        return len(next(iter(self._track_prop_dict.values())))

    # DATA

    def get_data(self, track_properties=None, normalize=True):
        """Returns data corresponding to the given data properties as
        a tensorflow array. By default, returns non-normalized active
        data.

        Args:
            track_properties: a list of track properties. If None,
                returns the active data.
            normalize: normalize the data within each track property.
                False by default.

        Returns:
            A tensor array of this dataset's data. It is indexed on the
            first axis by track number, and on the second by track
            property.

        Raises:
            ValueError: if one of the given track properties is not in
                this dataset.
        """

        if track_properties is None:
            track_properties = self.get_active_data_properties()
        else:
            for track_property in track_properties:
                if track_property not in self.get_available_data_properties():
                    raise ValueError("Provided track property {} not available"
                                     "in this dataset.".format(track_property))

        return tftrans(tfconst(list(map(lambda track_property:
                    ndops.normalize_val_list(
                        self._track_prop_dict[track_property]) if normalize\
                                else self._track_prop_dict[track_property],
                    track_properties)), dtype=float64))

    def get_active_data_properties(self):
        """Returns a list of the current active data properties in this
        TrackPropertiesDataset."""

        return self._active_data_properties

    def get_available_data_properties(self):
        """Returns a list of all data properties that this dataset has
        available. Note that the label property is an available data
        property."""

        return list(self._track_prop_dict.keys())

    def set_active_data_properties(self, track_properties=None):
        """Sets the given track properties as the active data properties
        within this dataset. Note that the label property is not barred
        from being a data property.

        Args:
            track_properties: a list of track properties, or None. If
                None, all available track properties become active data
                properties.

        Raises:
            ValueError: if this dataset does not have a data property
                corresponding to each element of the given list.
        """

        for track_property in track_properties:
            if track_property not in self.get_available_data_properties():
                raise ValueError("Provided track property {} not available"
                                 "in this dataset.".format(track_property))

        self._active_data_properties = track_properties

    # LABELS

    def get_labels(self):
        """Return a tensor list of this dataset's labels."""

        return tfconst(self._track_prop_dict[self._label_property])

    def get_label_property(self):
        """Return this dataset's label property."""

        return self._label_property

    def set_label_property(self, track_property):
        """Sets this dataset's label property.

        Args:
            track_property: a track property.

        Raises:
            ValueError: if the given track property is not in this
                dataset.
        """

        if track_property not in self.get_available_data_properties():
            raise ValueError("Provided track property " + track_property +
                             "not found in this dataset.")

        self._label_property = track_property

    # PREDICTIONS

    def get_all_predictions(self):
        """Returns a dict from label names to lists of predicted
        labels."""

        return dict(map(lambda pred_name:
                        (pred_name, self.get_prediction(pred_name)),
                        self.get_all_prediction_names()))

    def get_all_prediction_names(self):
        """Returns a list of names of all predictions in this
        dataset."""

        return list(self._predictions.keys())

    def get_prediction(self, pred_name):
        """Returns a list of predicted labels as a list.

        Args:
            pred_name: the name of a prediction.

        Returns:
            A list of predicted labels.

        Raises:
            ValueError: if the given pred_name does not correspond to a
                prediction in this TrackPropertiesDataset.
        """

        if pred_name not in self.get_all_prediction_names():
            raise ValueError("{} is not a prediction in this dataset."
                             .format(pred_name))

        return self._predictions[pred_name]

    def add_prediction(self, pred_name, pred_labels):
        """Adds a list of predictions to this model accessible by a
        name. The predictions must be generated from the data already
        present.

        Args:
            pred_name: a name by which to reference these predictions.
            pred_labels: the predicted labels. They can be probablistic
                or one hot, but some plotting functionality will only
                work for probablistic predictions.
        """

        if len(pred_labels) != self.size():
            raise ValueError("Given prediction list has different number of "
                             "elements ({}) than the number in this dataset ({})"
                             .format(len(pred_labels), len(self.size())))

        self._predictions.update({pred_name: pred_labels})

    def remove_prediction(self, pred_name):
        """Removes a prediction from this dataset's prediction dict."""

        self._predictions.pop(pred_name)

    # CUTS

    def cut(self, selector_dict):
        """Cuts this dataset by track property values, returns the cut
        dataset.

        Args:
            selector_dict: a track property dict selector.

        Returns:
            A cut TrackPropertiesDataset.
        """

        indices_to_cut = ndops.select_indices(
            self._track_prop_dict, selector_dict)
        return TrackPropertiesDataset(
            ndops.cut_track_prop_dict_by_indices(
                self._track_prop_dict, indices_to_cut),
            self.get_label_property(),
            self.get_active_data_properties(),
            ndops.cut_track_prop_dict_by_indices(
                self._predictions, indices_to_cut))

    def split(self, split_list):
        """Returns datasets of number and relative sizes of elements as
        specified in split_dist. Retains the same active properties and
        label property. Does not alter the calling dataset, includes
        predictions.

        Args:
            split_list: a list of relative sizes of output datasets.
                For example, [.7 .2 .1] and [70 20 10] will produce the
                same output.

        Returns:
            A list of TrackPropertiesDatasets.
        """

        split_tpds = ndops.split_track_prop_dict(self._track_prop_dict,
                                                 split_list)

        split_preds = ndops.split_track_prop_dict(self._predictions,
                                                  split_list)

        return list(map(lambda split_tpd, split_preds:
                        TrackPropertiesDataset(split_tpd,
                                               self.get_label_property(),
                                               self.get_active_data_properties(),
                                               split_preds),
                        split_tpds, split_preds))

    # OTHER

    def to_track_prop_dict(self, include_preds=False):
        """Converts this TrackPropertiesDataset to a track properties
        dict, with or without labeled predictions. Includes all
        available track properties, not just active data properties.

        Args:
            include_preds: if True, include predictions in the returned
            track properties dict.

        Returns:
            A track properties dict.
        """

        if include_preds:
            tpd_to_return = deepcopy(self._track_prop_dict)
            tpd_to_return.update(deepcopy(self._predictions))
            return tpd_to_return
        else:
            return deepcopy(self._track_prop_dict)
