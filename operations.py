from random import shuffle
from random import seed as set_seed
from copy import deepcopy
from functools import reduce


def ntuples_to_ntuple_dict(event_sets, properties_by_track_type):
    """Takes in a collection of uproot ntuples and a dictionary from
    track types to desired properties to be included, returns an ntuple
    dictionary formed by selecting properties from the ntuples and then
    concatenating them all together.

    Args:
        event_sets: a collection of uproot ntuples
        properties_by_track_type: a dictionary from track types (trk,
            matchtrk, etc.) to properties to be selected (eta, pt, chi2)

    Returns:
        An ntuple dictionary
    """

    return add_ntuple_dicts(list(map(lambda event_set: 
        ntuple_to_ntuple_dict(event_set, properties_by_track_type), 
        event_sets)))


def add_ntuple_dicts(ntuple_dicts):
    """Adds together multiple ntuple dicts of with the same track types and
    track type properties. Raises an exception if the dicts do not have this
    "sameness" property.

    Args:
        ntuple_dicts: a list of ntuple dicts with the same track types and
            track type properties

    Returns:
        An ntuple dictionary with the lists of values of each ntuple dict in
        the input list concatenated
    """
    
    track_types = iter(next(iter(ntuple_dicts)).keys())

    return dict(map(lambda track_type:
        (track_type, add_track_prop_dicts(
            list(map(lambda ntuple_dict: ntuple_dict[track_type],
                ntuple_dicts)))),
        track_types))


def add_track_prop_dicts(track_prop_dicts):
    """Adds together multiple track properties dicts of with the same properties.
    Raises an exception if the dicts do not have this "sameness" property.

    Args:
        track properties_dicts: a list of track properties dicts with the
        same properties

    Returns:
        An track properties dictionary with the lists of values of each track
        properties dictionary in the input list concatenated
    """

    def add_two_track_prop_dicts(tp_so_far, tp_to_add):
        """Adds two track properties dicts together as per rules in parent function.
        Returns the sum."""

        return dict(map(lambda property, vals_so_far, vals_to_add:
            (property, vals_so_far + vals_to_add),
            tp_so_far.keys(), list(tp_so_far.values()), list(tp_to_add.values())))

    return reduce(add_two_track_prop_dicts, track_prop_dicts)


def ntuple_to_ntuple_dict(event_set, properties_by_track_type):
    """Turns an uproot ntuple into an ntuple dictionary.

    Args:
        event_set: an uproot ntuple
        properties_by_track_type: a dictionary from track types (trk,
             matchtrk, etc.) to properties to be selected (eta, pt, chi2)

    Returns:
        An ntuple dictionary
    """

    return dict(map(lambda track_type, properties: 
        (track_type, ntuple_to_track_prop_dict(event_set, track_type, properties)),
        properties_by_track_type.keys(), properties_by_track_type.values()))


def ntuple_to_track_prop_dict(event_set, track_type, properties):
    """Takes in an uproot ntuple, the data type, and properties to be extracted;
    returns a dictionary from a property name to flattened array of values.
    Note that due to this flattening, all information about which tracks are
    from which event is lost.

    Args:
        event_set: an uproot event set
        track_type: trk, matchtrk, etc.
        properties: pt, eta, pdgid, etc.

    Returns:
        A tracks properties dictionary
    """

    def get_property_list(property):
        """Returns the list of properties corresponding to the event set,
        track type, and property name."""

        return list(event_set[track_type + "_" + property].array().flatten())

    return dict(map(lambda property: (property, get_property_list(property)),
        properties))


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
    not all of the same length."""

    # A fancy way of checking if all value lists are the same length
    val_list_lengths = set(map(len, track_prop_dict.values()))
    if len(val_list_lengths) > 1:
        raise ValueError("Invalid track prop dictionary:"
                "value lists are of different sizes")

    return next(iter(val_list_lengths)) 


def shuffle_ntuple_dict(ntuple_dict, seed=None):
    """Returns an ntuple dict whose value lists have been shuffled. To
    preserve association between them, value lists of trk and matchtp
    as well as ones for tp and matchtrk have been shuffled in the same
    way.

    Args:
        ntuple_dict: an ntuple dictionary
        seed: a seed for the random shuffling for reproducability

    Returns:
        An ntuple dict with its value lists shuffled, preserving the
        association between complementary track types.
    """

    #FIXME I hate all this hardcoding but I can't see any other way to do it

    # Generate shuffled indices dictionary
    ntuple_dict_num_tracks = ntuple_dict_length(ntuple_dict)
    shuffled_indices_dict = {"trk": [], "matchtrk": [], "tp": [], "matchtp": []}
    set_seed(seed)

    def generate_shuffled_indices_dict_pair(track_type, track_prop_dict):
        """Generates a pair to be used in the construction of a
        shuffled indices dictionary."""

        tpd_indices = list(range(track_prop_dict_length(track_prop_dict)))
        shuffle(tpd_indices)

        return (track_type, tpd_indices)

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
        track_prop_dict: a track properties dictionary
        shuffled_indices: a complete list of indices in the range of
            the number of tracks in this track properties dict. Used
            to completely determine a shuffling
        seed: a seed for the random shuffling for reproducability

    Returns:
        A track properties dict whose value lists have been shuffled.

    Raises:
        ValueError: if shuffled_indices is different length than
        track_prop_dict
    """
    
    def generate_shuffled_indices(tpd_length):
        """Generates a list of shuffled indices for use in shuffling
        tracks in this track property dictionary."""

        tpd_indices = list(range(tpd_length))
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
        raise ValueError("shuffled_indices length differs from"
                "track_prop_dict length")
    
    return dict(map(lambda property_name, val_list:
        (property_name, shuffle_val_list(val_list, shuffled_indices)),
        track_prop_dict.keys(), track_prop_dict.values()))


def reduce_ntuple_dict(ntuple_dict, track_limit=10,
        shuffle_tracks=True, seed=None):
    """Reduces an ntuple dictionary to a number of tracks. If number of tracks
    in the ntuple is less than the track limit specified, print all tracks.
    Can be used for convenient print debugging. Does not affect the original
    ntuple dictionary.

    Args:
        ntuple_dict: an ntuple dictionary
        track limit: number of tracks to retain in each value list. Or, an
            integer that will be expanded into a corresponding dictionary
        shuffle_tracks: if True, shuffles the value lists before reducing
        seed: a seed for the shuffling, for reproducability

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
            track_prop_dict, track_limit[track_type])),
        ntuple_dict.keys(), ntuple_dict.values()))


def reduce_track_prop_dict(track_prop_dict, track_limit=10,
        shuffle_tracks=True, seed=None):
    """Reduces a track properties dictionary such that each of its value
    lists are only a certain length. Does not affect the original track
    property dictionary.

    Args:
        track_prop_dict: a track properties dictionary
        track_limit: the maximum length for a value list
        shuffle_tracks: if True, shuffles the value lists before reducing
        seed: a seed for the shuffling, for reproducability

    Returns:
        A track properties dictionary with reduced-length value lists.
    """

    if shuffle_tracks:
        track_prop_dict = shuffle_track_prop_dict(track_prop_dict, seed)

    return dict(map(lambda track_prop, track_prop_vals:
        (track_prop, track_prop_vals[:min(track_limit, len(track_prop_vals))]),
        track_prop_dict.keys(), track_prop_dict.values()))


def select(*selector_key):
    """Takes in a selector key and returns a selector that returns true for
    selected values and false for non-selected values. This is how cuts are
    applied in this setup.
    
    Args:
        selector_key: If a single number, the selector will return true for
            that number. If two numbers, the selector will return true for
            numbers in that range, inclusive.

    Returns:
        A selector, a function that returns true for some numbers and false
        for others.

    Raises:
        ValueError: for invalid selector keys
    """

    if len(selector_key) == 1:
        return lambda val: val == next(iter(selector_key))
    if len(selector_key) == 2:
        return lambda val: val >= selector_key[0] and val <= selector_key[1]
    else:
        raise ValueError("Invalid selector key: {}. Read the docs!"
                .format(selector_key))


def cut_ntuple(ntuple_dict, nd_selector={}):
    """Cuts an ntuple dictionary by cutting each track type according to a
    selector dictionary, cutting those tracks not selected. Tracks are cut
    "symmetrically" across corresponding groups, meaning that any cuts applied
    to trks are applied to matchtps, and from tps to matchtrks, and vice versa.

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
                                     nd_selector.keys(), nd_selector.values())))

    # Combine trk and matchtp, tp and matchtrk indices
    # Sort and remove duplicates
    trk_matchtp_indices_to_cut = sorted(
        list(dict.fromkeys(cut_indices_dict["trk"] + cut_indices_dict["matchtp"])))
    tp_matchtrk_indices_to_cut = sorted(
        list(dict.fromkeys(cut_indices_dict["tp"] + cut_indices_dict["matchtrk"])))

    cut_ntuple_dict = {}
    for track_type, trackset in ntuple_dict.items():
        if track_type in ["trk", "matchtp"]:
            indices_to_cut = trk_matchtp_indices_to_cut
        if track_type in ["tp", "matchtrk"]:
            indices_to_cut = tp_matchtrk_indices_to_cut
        cut_ntuple_dict[track_type] = cut_trackset_by_indices(
            ntuple_dict[track_type], indices_to_cut)

    return cut_ntuple_dict


def cut_trackset(track_prop_dict, tpd_selector={}):
    """Cuts an track properties dictionary by cutting each track type according
    to a cut dictionary.

    Args:
        track_prop_dict: a tracks properties dictionary
        tpd_selector: a selector for a tracks properties dictionary

    Returns:
        A cut tracks properties dictionary
    """

    return cut_trackset_by_indices(track_prop_dict,
            select_indices(track_prop_dict, tpd_selector))


def select_indices(track_prop_dict, tpd_selector, invert=True):
    """Selects indices from a tracks properties dictionary that meet the
    conditions of the selector dictionary. If a property is in the selector
    dict but not in the tracks properties dict, the program won't raise an
    exception, but will print a message.

    Args:
        track_prop_dict: a tracks properties dictionary
        tpd_selector: a dictionary from track property names to selectors
        inverse: return all indices NOT selected. Default is True as this
            jibes with how this function is mainly used: track cuts

    Returns:
        Indices from the track properties dict selected by the selector dict
    """

    # Determine which selection conditions will be applied
    for property in tpd_selector.keys():
        if property not in track_prop_dict.keys():
            print(property + " not in tracks properties; will not select")
            tpd_selector.pop(property)

    def index_meets_selection(track_index):
        """Determine if the track at this index is selected by the selector
        dict."""

        return all(list(map(lambda track_property, property_selector:
            property_selector(track_prop_dict[track_property][track_index]),
            tpd_selector.keys(), tpd_selector.values())))

    track_indices = range(len(next(iter(track_prop_dict.values()))))
    return list(filter(lambda track_index:
            invert != index_meets_selection(track_index),
            track_indices))


def cut_trackset_by_indices(track_prop_dict, indices_to_cut):
    """Takes in a list of indices to cut and cuts those indices from the
    lists of the dictionary. Assumes that all lists in track_prop_dict
    are the same size. This list of indices will frequently be generated
    using get_indices_meeting_condition. The list of indices does not
    have to be sorted by size.

    Args:
        track_prop_dict: a tracks properties dictionary
        indices_to_cut: a collection of indices to cut. Repeats are
                tolerated, but out-of-range indices will result in an
                exception.

    Returns:
        The same tracks properties dictionary with the given indices
        on its value lists removed.
    """

    # Copy, then delete all tracks at indices (backwards)
    cut_track_prop_dict = deepcopy(track_prop_dict)
    for track_to_cut in reversed(sorted(indices_to_cut)):
        for property in cut_track_prop_dict.keys():
            del cut_track_prop_dict[property][track_to_cut]

    return cut_track_prop_dict


def get_proportion_selected(tracks_property, selector, norm=True):
    """Find the proportion of tracks selected with the given selector.
    If there are no tracks in the tracks property value list, returns zero.
    Can also return the number of tracks meeting the condition.

    Args:
            tracks_property: a list of values of a track property, such as
                    trk_pt or tp_chi2rphi
            property_condition: a property that these value can satisfy. For
                    example, "lambda trk_eta: trk_eta <= 2.4".
            norm: if True, divides the number of tracks meeting the condition
                    by the total number of tracks. This is the default option.

    Returns:
            Either the number or proportion of tracks meeting the condition,
            depending on the value of norm.
    """

    if len(tracks_property) == 0:
        print("Cannot calculate proportion meeting condition in zero-length"
                "quantity. Returning zero.")
        return 0

    num_tracks_meeting_cond = sum(map(selector, tracks_property))
    return float(num_tracks_meeting_cond) / len(tracks_property) if norm \
            else num_tracks_meeting_cond

