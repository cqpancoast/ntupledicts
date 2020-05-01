from ntupledicts import *
from copy import deepcopy
from functools import reduce


def ntuples_to_ntuple_dict(event_sets, properties_by_track_type):
    """Takes in a collection of uproot ntuples and a dictionary from
    track types to desired properties to be included, returns an ntuple
    dictionary formed by selecting properties from the ntuples and then
    concatenating them all together.

    Args:
        event_sets:  a collection of uproot ntuples
        properties_by_track_type:  a dictionary from track types (trk,
            matchtrk, etc.) to properties to be selected (eta, pt, chi2)

    Returns:
        An ntuple dictionary
    """

    return add_ntuple_dicts(map(lambda event_set: 
        ntuple_to_ntuple_dict(event_set, properties_by_track_type), 
        event_sets))


def ntuple_to_ntuple_dict(event_set, properties_by_track_type):
    """Turns an uproot ntuple into an ntuple dictionary.

    Args:
        event_set:  an uproot ntuple
        properties_by_track_type:  a dictionary from track types (trk,
             matchtrk, etc.) to properties to be selected (eta, pt, chi2)

    Returns:
        An ntuple dictionary
    """

    return dict(map(lambda track_type, properties: 
        (track_type, ntuple_to_tracks_prop_dict(event_set, track_type, properties)),
        properties_by_track_type.keys(), properties_by_track_type.values()))


def ntuple_to_tracks_prop_dict(event_set, track_type, properties):
    """Takes in an uproot ntuple, the data type, and properties to be extracted;
    returns a dictionary from a property name to flattened array of values.
    Note that due to this flattening, all information about which tracks are
    from which event is lost.

    Args:
            event_set:  an uproot event set
            track_type:  trk, matchtrk, etc.
            properties:  pt, eta, pdgid, etc.

    Returns:
            A tracks properties dictionary
    """

    def get_property_list(property):
        """Returns the list of properties corresponding to the event set,
        track type, and property name."""

        return list(event_set[track_type + "_" + property].array().flatten())

    return dict(map(lambda property: (property, get_property_list(property)),
        properties))


def add_ntuple_dicts(ntuple_dicts):
    """Adds together multiple ntuple dicts of with the same track types and
    track type properties. Raises an exception if the dicts do not have this
    "sameness" property.

    Args:
        ntuple_dicts:  a list of ntuple dicts with the same track types and
            track type properties

    Returns:
        An ntuple dictionary with the lists of values of each ntuple dict in
        the input list concatenated
    """

    return None


def add_track_prop_dicts(track_prop_dicts):
    """Adds together multiple track properties dicts of with the same properties.
    Raises an exception if the dicts do not have this "sameness" property.

    Args:
        track properties_dicts:  a list of track properties dicts with the
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
            tp_so_far.keys(), list(tp_so_far.values()), list(tp_to_add.values()))

    return reduce(add_two_track_prop_dicts, track_prop_dicts)


def cut_ntuple(ntuple_properties,
               cut_dicts={"tp": {"eta": [-2.4, 2.4], "pt": [2, 13000], "nstub": [4, 999]},
                          "trk": {"eta": [-2.4, 2.4], "pt": [2, 13000], "nstub": [4, 999]}}):
    """Takes in this file's representation of an ntuple (a dict from track types
    to dicts from properties to value lists) and cuts each trackset. Takes into
    account that matchtrks and tps must be cut together, and that trks and matchtps
    must be cut together as well."""

    # Build list of tracks to cut from tp/matchtrk group and trk/matchtp groups
    cut_indices_dict = {"trk": [], "matchtrk": [], "tp": [], "matchtp": []}
    cut_indices_dict.update(dict(map(lambda track_type, cut_dict:
                                     (track_type, get_indices_meeting_conditions(
                                         ntuple_properties[track_type], cut_dict)),
                                     cut_dicts.keys(), cut_dicts.values())))

    # Combine trk and matchtp, tp and matchtrk indices to remove (respectively), sorting and removing duplicates
    trk_matchtp_indices_to_cut = sorted(
        list(dict.fromkeys(cut_indices_dict["trk"] + cut_indices_dict["matchtp"])))
    tp_matchtrk_indices_to_cut = sorted(
        list(dict.fromkeys(cut_indices_dict["tp"] + cut_indices_dict["matchtrk"])))

    cut_ntuple_properties = {}
    for track_type, trackset in ntuple_properties.items():
        if track_type in ["trk", "matchtp"]:
            indices_to_cut = trk_matchtp_indices_to_cut
        if track_type in ["tp", "matchtrk"]:
            indices_to_cut = tp_matchtrk_indices_to_cut
        cut_ntuple_properties[track_type] = cut_trackset_by_indices(
            ntuple_properties[track_type], indices_to_cut)

    return cut_ntuple_properties


def cut_trackset(tracks_properties, cut_dict={"eta": [-2.4, 2.4], "pt": [2, 13000], "nstub": [4, 999]}):
    """Cuts a trackset (trk_XYZ, tp_XYZ, etc.) given the cut dicts."""

    return cut_trackset_by_indices(tracks_properties,
                                   get_indices_meeting_conditions(tracks_properties, cut_dict))


def get_indices_meeting_conditions(tracks_properties, cond_dict={}):
    """Takes in a dictionary from track properties to a list of corresponding
    values and conditions to select tracks from the set. If a value from the cond dict
    is not in the track properties, the program will not select, but will report it.
    Selection conditions are the name of a property, and then either:
            - a list containing a lower and upper bound, both inclusive.
            - a single number indicating a value to be selected.

    For a selection to take place, the property value lists in tracks_properties must
    all be the same size. This function asserts that this is the case.

    Returns a list of indices that meet the conditions.
    """

    # Assert that all property lists are the same size
    supposed_num_tracks = len(next(iter(tracks_properties.values())))
    for track_property_values in tracks_properties.values():
        assert(len(track_property_values) == supposed_num_tracks)
    num_tracks = supposed_num_tracks

    # Determine which selection conditions will be applied
    for property in cond_dict.keys():
        if property not in tracks_properties.keys():
            print(property + " not in tracks properties; will not select")
            cond_dict.pop(property)

    def select_track_property(track_property, cond):
        """Returns whether this track property needs to be selected."""

        if isinstance(cond, list):
            return track_property < cond[0] or track_property > cond[1]
        else:
            return track_property != cond

    # Build up a list of indices of tracks to select
    tracks_to_select = []
    for track_index in range(num_tracks):
        for property, cond in cond_dict.items():
            if select_track_property(tracks_properties[property][track_index], cond):
                tracks_to_select.append(track_index)
                break

    return tracks_to_select


def cut_trackset_by_indices(tracks_properties, indices_to_cut):
    """Takes in a list of indices to cut and cuts those indices from the
    lists of the dictionary. Assumes that all lists in tracks_properties
    are the same size. This list of indices will frequently be generated
    using get_indices_meeting_condition. The list of indices does not
    have to be sorted by size.

    Args:
            tracks_properties:  a dictionary from track property names to
                    lists of track property values.
            indices_to_cut:  a collection of indices to cut. Repeats are
                    tolerated, but out-of-range indices will result in an
                    exception.

    Returns:
            A trackset of the same form as the input, but with the indices
            on its value lists removed.
    """

    # Copy, then delete all tracks (going backwards so as not to affect indices)
    cut_tracks_properties = deepcopy(tracks_properties)
    for track_to_cut in reversed(sorted(indices_to_cut)):
        for property in cut_tracks_properties.keys():
            del cut_tracks_properties[property][track_to_cut]

    return cut_tracks_properties

