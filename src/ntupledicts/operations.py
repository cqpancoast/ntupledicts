"""Basic operations on ntuple dicts and track property dicts."""

from random import shuffle
from random import seed as set_seed
from copy import deepcopy
from functools import reduce
from math import inf
from warnings import warn
from numpy import cumsum
from numpy import array
from numpy import delete
from numpy import where


def add_ntuple_dicts(ntuple_dicts):
    """Adds together multiple ntuple dicts of with the same track types
    and track type properties. Raises an exception if the dicts do not
    have this "sameness" property.

    Args:
        ntuple_dicts: a list of ntuple dicts with the same track types
            and track type properties.

    Returns:
        An ntuple dictionary with the lists of values of each ntuple
        dict in the input list concatenated.
    """

    track_types = iter(next(iter(ntuple_dicts)).keys())

    return dict(map(lambda track_type:
        (track_type, add_track_prop_dicts(
            list(map(lambda ntuple_dict: ntuple_dict[track_type],
                ntuple_dicts)))),
        track_types))


def add_track_prop_dicts(track_prop_dicts):
    """Adds together multiple track properties dicts of with the same
    properties.

    Args:
        track properties_dicts: a list of track properties dicts with
            the same properties.

    Returns:
        A track properties dictionary with the lists of values of each
        track properties dict in the input list concatenated.

    Raises:
        ValueError: if there is no property shared by all of the track
            property dicts.
    """

    def add_two_track_prop_dicts(tp_so_far, tp_to_add):
        """Adds two track properties dicts together as per rules in
        parent function. Returns the sum."""

        props_in_common = set(tp_so_far.keys()).intersection(
                set(tp_to_add.keys()))

        if props_in_common != set(tp_to_add.keys()):
            warn("Track property dicts have differing value lists. "
                    "Will add only properties in common: {}"
                    .format(props_in_common), UserWarning)

        if not len(props_in_common):
            raise ValueError("Track property dicts to add have no properties "
                    "in common.")

        return dict(map(lambda track_property:
            (track_property,
                tp_so_far[track_property] + tp_to_add[track_property]),
            props_in_common))

    return reduce(add_two_track_prop_dicts, track_prop_dicts)


def mix_track_prop_dicts(track_prop_dicts, seed=None):
    """Mixes together multiple track properties dicts with the same
    properties. 'Mixing', in this context, means to cut all other dicts
    in the list down to the size of the smallest, and then shuffle them
    all together.

    Args:
        track properties_dicts: a list of track properties dicts with
            the same properties.
        seed: a seed for the random shuffling for reproducability.

    Returns:
        A track properties dictionary containing an equal amount of
        tracks from each track properties dictionary in the input list.

    Raises:
        ValueError: if there is no property shared by all of the track
            property dicts.
    """

    min_tpd_size = min(map(track_prop_dict_length, track_prop_dicts))

    return shuffle_track_prop_dict(add_track_prop_dicts(map(
            lambda tpd: reduce_track_prop_dict(tpd, min_tpd_size, seed),
            track_prop_dicts)),
        seed=seed)


def ntuple_dict_length(ntuple_dict):
    """Returns a dictionary from track types to the number of tracks of
    that type. Raises an exception of any value lists within one of its
    track properties dicts are different lengths."""

    return dict(map(lambda track_type, track_prop_dict:
        (track_type, track_prop_dict_length(track_prop_dict)),
        ntuple_dict.keys(), ntuple_dict.values()))


def track_prop_dict_length(track_prop_dict):
    """Returns the number of tracks in a track properties dictionary.
    Raises an exception if the value lists in the input dictionary are
    not all of the same length. Returns zero if the track properties
    dict is empty."""

    # A fancy way of checking if all value lists are the same length
    val_list_lengths = set(map(len, track_prop_dict.values()))
    if len(val_list_lengths) > 1:
        raise ValueError("Invalid track prop dictionary: "
                "value lists are of different sizes")
    if len(val_list_lengths) == 0:
        return 0

    return next(iter(val_list_lengths))


class TrackPropertyDictIterator:
    """Iterates through tracks in a track properties dict, where each
    track is represented as a dictionary from a value name to a single
    property. Does not alter the original track properties dict."""

    def __init__(self, track_prop_dict, increment=1):
        self.tpd = deepcopy(track_prop_dict)
        self.track_index = -1
        self.increment = increment
        self.num_tracks = track_prop_dict_length(track_prop_dict)

    def __iter__(self):
        return self

    def __next__(self):
        self.track_index += self.increment
        if self.track_index < self.num_tracks:
            return self._grab_track_by_index(self.tpd, self.track_index)
        raise StopIteration

    def _grab_track_by_index(self, track_prop_dict, track_index):
        """Returns a dictionary from track properties to single values
        by selecting that value from each list in the track properties
        dict."""

        return dict(map(lambda property_name, val_list:
            (property_name, val_list[track_index]),
            track_prop_dict.keys(), track_prop_dict.values()))


def shuffle_ntuple_dict(ntuple_dict, seed=None):
    """Returns an ntuple dict whose value lists have been shuffled. To
    preserve association between them, value lists of trk and matchtp
    as well as ones for tp and matchtrk have been shuffled in the same
    way.

    Args:
        ntuple_dict: an ntuple dictionary.
        seed: a seed for the random shuffling for reproducability.

    Returns:
        An ntuple dict with its value lists shuffled, preserving the
        association between complementary track types.
    """

    # Generate shuffled indices dictionary
    ntuple_dict_num_tracks = ntuple_dict_length(ntuple_dict)
    shuffled_indices_dict = {"trk": [], "matchtrk": [], "tp": [], "matchtp": []}
    set_seed(seed)

    def generate_shuffled_indices_dict_pair(track_type, track_prop_dict):
        """Generates a pair to be used in the construction of a
        shuffled indices dictionary."""

        tpd_indices = list(range(track_prop_dict_length(track_prop_dict)))
        shuffle(tpd_indices)

        return track_type, tpd_indices

    shuffled_indices_dict.update(dict(map(generate_shuffled_indices_dict_pair,
        ntuple_dict.keys(), ntuple_dict.values())))

    # Ensure that the ntuple dict num tracks dict has the appropriate
    # number of keys
    ntuple_dict_num_tracks.update(dict(map(lambda track_type, indices:
        (track_type, len(indices)),
        shuffled_indices_dict.keys(), shuffled_indices_dict.values())))

    # Ensure that same-length, complementary track types shuffle the same
    if ntuple_dict_num_tracks["trk"] == ntuple_dict_num_tracks["matchtp"]:
        shuffled_indices_dict["trk"] = shuffled_indices_dict["matchtp"]
    if ntuple_dict_num_tracks["matchtrk"] == ntuple_dict_num_tracks["tp"]:
        shuffled_indices_dict["matchtrk"] = shuffled_indices_dict["tp"]

    return dict(map(lambda track_type, track_prop_dict:
        (track_type, shuffle_track_prop_dict(
            track_prop_dict, shuffled_indices_dict[track_type], seed)),
        ntuple_dict.keys(), ntuple_dict.values()))


def shuffle_track_prop_dict(track_prop_dict, shuffled_indices=None, seed=None):
    """Returns a track properties dict whose value lists have been
    shuffled.

    Args:
        track_prop_dict: a track properties dictionary.
        shuffled_indices: a complete list of indices in the range of
            the number of tracks in this track properties dict. Used
            to completely determine a shuffling.
        seed: a seed for the random shuffling for reproducability.

    Returns:
        A track properties dict whose value lists have been shuffled.

    Raises:
        ValueError: if shuffled_indices is different length than
        track_prop_dict.
    """

    def generate_shuffled_indices(tpd_length):
        """Generates a list of shuffled indices for use in shuffling
        tracks in this track property dictionary."""

        tpd_indices = list(range(tpd_length))
        set_seed(seed)
        shuffle(tpd_indices)

        return tpd_indices

    def shuffle_val_list(val_list, shuffled_indices):
        """Shuffles a value list depending on whether there are shuffled
        indices or a random seed provided."""

        return list(map(lambda i: val_list[i], shuffled_indices))

    tpd_length = track_prop_dict_length(track_prop_dict)

    if shuffled_indices is None:
        shuffled_indices = generate_shuffled_indices(tpd_length)
    if len(shuffled_indices) != tpd_length:
        raise ValueError("shuffled_indices arg length ({}) differs from "
                "track_prop_dict length ({})."
                .format(len(shuffled_indices), tpd_length))

    return dict(map(lambda property_name, val_list:
        (property_name, shuffle_val_list(val_list, shuffled_indices)),
        track_prop_dict.keys(), track_prop_dict.values()))


def reduce_ntuple_dict(ntuple_dict, track_limit,
                       shuffle_tracks=False, seed=None):
    """Reduces an ntuple dictionary to a number of tracks. If number of
    tracks in the ntuple is less than the track limit specified, returns
    all tracks. Can be used for convenient print debugging. Does not
    affect the original ntuple dictionary.

    Args:
        ntuple_dict: an ntuple dict.
        track_limit: number of tracks to retain in each value list. Or,
            an integer that will be expanded into a corresponding
            dictionary.
        shuffle_tracks: if True, shuffles the value lists before
            reducing.
        seed: a seed for the shuffling, for reproducability.

    Returns:
        An ntuple dictionary with track_limit tracks.
    """

    # Get track_limit into correct form if it's an int
    if isinstance(track_limit, int):
        track_limit = dict(map(lambda track_type:
            (track_type, track_limit),
            ntuple_dict.keys()))

    if shuffle_tracks:
        ntuple_dict = shuffle_ntuple_dict(ntuple_dict, seed)

    return dict(map(lambda track_type, track_prop_dict:
        (track_type, reduce_track_prop_dict(
            track_prop_dict, track_limit[track_type], shuffle_tracks=False)),
        ntuple_dict.keys(), ntuple_dict.values()))


def reduce_track_prop_dict(track_prop_dict, track_limit, min_index=0,
                           shuffle_tracks=True, seed=None):
    """Reduces a track properties dictionary such that each of its value
    lists are only a certain length. Does not affect the original track
    property dictionary.

    Args:
        track_prop_dict: a track properties dict.
        track_limit: the maximum length for a value list.
        min_index: the first index to include in the result.
        shuffle_tracks: if True, shuffles the value lists before
            reducing.
        seed: a seed for the shuffling, for reproducability.

    Returns:
        A track properties dictionary with reduced-length value lists.
    """

    if shuffle_tracks:
        track_prop_dict = shuffle_track_prop_dict(track_prop_dict, seed=seed)

    return dict(map(lambda track_prop, track_prop_vals:
        (track_prop, track_prop_vals[min_index:min(track_limit + min_index,
            len(track_prop_vals))]),
        track_prop_dict.keys(), track_prop_dict.values()))


def split_track_prop_dict(track_prop_dict, split_list):
    """Splits a track properties dict into smaller ones according to
    the relative sizes of split_list elements. There is no option to
    shuffle these, as the dict can be shuffled before splitting.

    Args:
        track_prop_dict: a track properties dict.
        split_list: a list of positive values that determine the number
            and relative sizes of the post-split track property dicts.

    Returns:
        A list of track property dicts.
    """

    def get_split_sizes(split_list, num_tracks):
        """Returns the sizes of data by normalizing the provided split
        distribution and mutliplying by the number of tracks in such
        a way that the resulting sizes add up to the original tracks."""

        split_list_total = sum(split_list)
        split_sizes = list(map(lambda split_val:
            int(split_val * num_tracks / split_list_total),
            split_list))

        # Ensure the split sizes add up to the total number of tracks
        split_sizes[-1] += num_tracks - sum(split_sizes)

        return split_sizes

    split_boundaries = [0] + list(cumsum(get_split_sizes(split_list,
            track_prop_dict_length(track_prop_dict))))

    return list(map(lambda start_index, end_index:
        reduce_track_prop_dict(track_prop_dict, end_index - start_index,
            start_index, shuffle_tracks=False),
        split_boundaries[:-1], split_boundaries[1:]))


def select(*selector_key, invert=False):
    """Takes in a selector key and returns a selector that returns true
    for selected values and false for non-selected values. This is how
    cuts are applied in ntupledicts.

    Args:
        selector_key: If a single number, the selector will return true
            for that number. If two numbers, the selector will return
            true for numbers in that range, inclusive. If a list, will
            treat each element of a list as a selector and logical OR
            them together. This is done instead of AND, as AND can
            simply be achieved by applying multiple selectors.
        invert: Invert the selection. False by default.

    Returns:
        A selector, a function that returns true for some values and
        false for all others.

    Raises:
        ValueError: for invalid selector keys.
    """

    if len(selector_key) == 1:
        key_contents = next(iter(selector_key))
        if isinstance(key_contents, list):
            selector = lambda val: any(map(
                lambda sub_selector: sub_selector(val),
                key_contents))
        elif isinstance(key_contents, (float, int)):
            selector = lambda val: val == next(iter(selector_key))
        else:
            raise ValueError("Invalid selector key type: {}."
                    .format(type(key_contents)))
    elif len(selector_key) == 2:
        selector = lambda val: selector_key[0] <= val <= selector_key[1]
    else:
        raise ValueError("Invalid selector key length: {}. Read the docs!"
                         .format(selector_key))

    return lambda val: invert != selector(val)


def cut_ntuple_dict(ntuple_dict, nd_selector):
    """Cuts an ntuple dictionary by cutting each track type according to
    a selector dictionary, cutting those tracks not selected. Tracks are
    cut "symmetrically" across corresponding groups, meaning that any
    cuts applied to trks are applied to matchtps, and from tps to
    matchtrks, and vice versa.

    Args:
        ntuple_dict: an ntuple dictionary
        nd_selector: a selector for an ntuple dict

    Returns:
        A cut ntuple dictionary
    """

    # Build list of tracks to cut from tp/matchtrk group and trk/matchtp groups
    cut_indices_dict = {"trk": [], "matchtrk": [], "tp": [], "matchtp": []}
    cut_indices_dict.update(dict(map(lambda track_type, cut_dict:
                                     (track_type, select_indices(
                                         ntuple_dict[track_type], cut_dict)),
                                     nd_selector.keys(),
                                     nd_selector.values())))

    # Combine trk and matchtp, tp and matchtrk indices
    # Sort and remove duplicates
    trk_matchtp_indices_to_cut = sorted(
        list(dict.fromkeys(cut_indices_dict["trk"] +\
                cut_indices_dict["matchtp"])))
    tp_matchtrk_indices_to_cut = sorted(
        list(dict.fromkeys(cut_indices_dict["tp"] +\
                cut_indices_dict["matchtrk"])))

    cut_ntuple_dict = {}
    for track_type in ntuple_dict.keys():
        if track_type in ["trk", "matchtp"]:
            indices_to_cut = trk_matchtp_indices_to_cut
        if track_type in ["tp", "matchtrk"]:
            indices_to_cut = tp_matchtrk_indices_to_cut
        cut_ntuple_dict[track_type] = cut_track_prop_dict_by_indices(
            ntuple_dict[track_type], indices_to_cut)

    return cut_ntuple_dict


def cut_track_prop_dict(track_prop_dict, tpd_selector):
    """Cuts an track properties dictionary by cutting each track type
    according to a cut dictionary.

    Args:
        track_prop_dict: a tracks properties dictionary.
        tpd_selector: a selector for a tracks properties dictionary.

    Returns:
        A cut tracks properties dictionary.
    """

    return cut_track_prop_dict_by_indices(track_prop_dict,
                select_indices(track_prop_dict, tpd_selector))


def select_indices(track_prop_dict, tpd_selector, invert=True):
    """Selects indices from a tracks properties dictionary that meet the
    conditions of the selector dictionary. If a property is in the
    selector dict but not in the tracks properties dict, the program
    won't raise an exception, but will print a message.

    Args:
        track_prop_dict: a tracks properties dictionary.
        tpd_selector: a dictionary from track property names to
            selectors.
        invert: return all indices NOT selected. Default is True. This
            jibes with how this function is mainly used: track cuts.

    Returns:
        Indices from the track properties dict selected by the selector
        dict.
    """

    # Determine which selection conditions will be applied
    for track_property in list(tpd_selector.keys()):
        if track_property not in track_prop_dict.keys():
            warn("{} not in tracks properties; will not select"
                    .format(track_property), UserWarning)
            tpd_selector.pop(track_property)

    tpd_length = track_prop_dict_length(track_prop_dict)
    return list(set(sum(map(lambda track_property, selector:
                            list(where([invert != selector(val) for val in\
                                track_prop_dict[track_property]])[0]),
                            tpd_selector.keys(), tpd_selector.values()),
                        [])))


def cut_track_prop_dict_by_indices(track_prop_dict, indices_to_cut):
    """Takes in a list of indices to cut and cuts those indices from the
    lists of the dictionary. Assumes that all lists in track_prop_dict
    are the same size. This list of indices will frequently be generated
    using get_indices_meeting_condition. The list of indices does not
    have to be sorted by size.

    Args:
        track_prop_dict: a tracks properties dictionary.
        indices_to_cut: a collection of indices to cut. Repeats are
            tolerated, but out-of-range indices will result in an
            exception.

    Returns:
        The same tracks properties dictionary with the given indices
        on its value lists removed.
    """

    track_properties = track_prop_dict.keys()
    post_cuts_track_prop_dict = {}
    for track_property in track_properties:
        post_cuts_track_prop_dict[track_property] = list(delete(
                array(track_prop_dict[track_property]), indices_to_cut))

    return post_cuts_track_prop_dict


def normalize_ntuple_dict(ntuple_dict, normalize_dict=None):
    """Normalizes each value list in an ntuple dict. Does not attempt
    to normalize values of the same property but different track types
    in the same way.

    Args:
        ntuple_dict: an ntuple dict.
        normalize_dict: a dictionary from track types to lists of
            properties to normalize of that track type. If None,
            normalizes all value lists.

    Returns:
        An ntuple dict with the appropriate value lists normalized.
    """

    base_normalize_dict = dict(map(lambda track_type: (track_type, None),
        ntuple_dict.keys()))
    base_normalize_dict.update(normalize_dict)

    return dict(map(lambda track_type:
        (track_type, normalize_track_prop_dict(ntuple_dict[track_type],
            base_normalize_dict[track_type])),
        base_normalize_dict.keys()))


def normalize_track_prop_dict(track_prop_dict, props_to_normalize=None):
    """Returns a track prop dict of the same form as the original, but
    each value list has been divided by its highest value. All values
    are normalized by default, but only some will be normalized if a
    list is given."""

    if props_to_normalize is None:
        props_to_normalize = list(track_prop_dict.keys())
    else:
        for track_property in props_to_normalize:
            if track_property not in list(track_prop_dict.keys()):
                warn("{} not in tracks properties; will not normalize"
                        .format(track_property), UserWarning)

    return dict(map(lambda track_property:
        (track_property, normalize_val_list(track_prop_dict[track_property])),
        props_to_normalize))


def normalize_val_list(val_list):
    """Returns a list of numeric values by the size of their maximum
    value."""

    max_val = float(max(val_list))

    return [ val/max_val if max_val != 0 else 0 for val in val_list ]

