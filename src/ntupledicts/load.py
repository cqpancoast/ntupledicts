"""LOAD: loads root files into ntuple dicts."""


from uproot import open as uproot_open
from . import operations as ndops
from .operations import select as sel


def root_files_to_ntuple_dict(root_ntuple_paths, properties_by_track_type,
        keep_invalid_vals=False):
    """The first function to be run in a typical ntupledicts session.

    Takes in paths to root track-trigger ntuples and a dict from
    track types to desired properties to be included, returns an ntuple
    dictionary formed by selecting properties from the ntuples and then
    concatenating them all together. Cuts any invalid values, like inf
    or nan, by default.

    Args:
        root_ntuple_paths: an iterable of paths to track-trigger root
            ntuples.
        properties_by_track_type: a dictionary from track types (trk,
            matchtrk, etc.) to properties to be selected
            (eta, pt, chi2).
        keep_invalid_vals: if True, don't cut tracks with inf or nan as
            one of their values.

    Returns:
        An ntuple dict, the star of the show.

    Raises:
        IOError: if any of the root files cannot be opened by uproot.
        IOError: if an event tree cannot be read from any opened uproot
            files.
    """

    # Extract uproot event trees
    uproot_ntuples = []
    for root_ntuple_path in root_ntuple_paths:

        # Open root files using uproot
        try:
            uprooted_file = uproot_open(root_ntuple_path)
        except:
            raise IOError("Root file at {} cannot be opened by uproot."
                    .format(root_ntuple_path))

        # Access event tree from opened uproot file
        try:
            uproot_event_set = next(iter(uprooted_file.values()))
            uproot_event_tree = next(iter(uproot_event_set.values()))
            uproot_ntuples.append(uproot_event_tree)
        except:
            raise IOError("Event tree cannot be read from opened file at {}."
                    .format(root_ntuple_path))

    return uproot_ntuples_to_ntuple_dict(uproot_ntuples,
            properties_by_track_type,
            keep_invalid_vals)


def uproot_ntuples_to_ntuple_dict(uproot_ntuples, properties_by_track_type,
        keep_invalid_vals=False):
    """Takes in a collection of uproot ntuples and a dictionary from
    track types to desired properties to be included, returns an ntuple
    dictionary formed by selecting properties from the ntuples and then
    concatenating them all together. Cuts any invalid values, like inf
    or nan, by default.

    Args:
        uproot_ntuples: an iterable of uproot ntuples.
        properties_by_track_type: a dictionary from track types (trk,
            matchtrk, etc.) to properties to be selected
            (eta, pt, chi2).
        keep_invalid_vals: if True, don't cut tracks with inf or nan as
            one of their values.

    Returns:
        An ntuple dict.
    """

    return ndops.add_ntuple_dicts(list(map(lambda uproot_ntuple:
        uproot_ntuple_to_ntuple_dict(uproot_ntuple,
            properties_by_track_type, keep_invalid_vals),
        uproot_ntuples)))


def uproot_ntuple_to_ntuple_dict(uproot_ntuple, properties_by_track_type,
        keep_invalid_vals=False):
    """Turns an uproot ntuple into an ntuple dictionary.

    Args:
        uproot_ntuple: an uproot ntuple.
        properties_by_track_type: a dictionary from track types (trk,
             matchtrk, etc.) to properties to be selected
             (eta, pt, chi2).
        keep_invalid_vals: if True, don't cut tracks with inf or nan as
            one of their values.

    Returns:
        An ntuple dict.
    """

    ntuple_dict = dict(map(lambda track_type, track_properties:
        (track_type, uproot_ntuple_to_track_prop_dict(
            uproot_ntuple, track_type, track_properties)),
        properties_by_track_type.keys(), properties_by_track_type.values()))

    if keep_invalid_vals:
        return ntuple_dict
    else:
        invalid_vals = [float("inf"), float("nan")]
        invalid_track_sel = lambda val: val not in invalid_vals

        # Select for the above selector in every field of an ntuple dict,
        # but only if the list contains those values in the first place
        invalid_sel_dict = dict(map(lambda track_type:
            (track_type, dict(map(lambda track_property:
                (track_property, invalid_track_sel),
                filter(lambda track_property:
                    any([invalid_val in ntuple_dict[track_type][track_property]\
                            for invalid_val in invalid_vals]),
                    ntuple_dict[track_type].keys()
                    )))),
            ntuple_dict.keys()))

        return ndops.cut_ntuple_dict(ntuple_dict, invalid_sel_dict)


def uproot_ntuple_to_track_prop_dict(uproot_ntuple, track_type,
        track_properties):
    """Takes in an uproot ntuple, the data type, and properties to be
    extracted; returns a dictionary from a property name to flattened
    array of values. Note that due to this flattening, all information
    about which tracks are from which event is lost.

    Args:
        uproot_ntuple: an uproot ntuple.
        track_type: a track type string. "trk", "matchtrk", etc.
        track_properties: a list of track property strings. "pt",
            "eta", "pdgid", etc.

    Returns:
        A track properties dict.
    """

    def get_value_list(track_property):
        """Returns the value list corresponding to uproot ntuple, track
        type, and property name."""

        return list(uproot_ntuple["{}_{}".format(track_type, track_property)]
                    .array().flatten())

    return dict(map(lambda track_property:
                    (track_property, get_value_list(track_property)),
                    track_properties))


