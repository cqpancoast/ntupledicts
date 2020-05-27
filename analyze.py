"""Analyzes the contents of ntuple dicts, track property dicts, and
value lists.

Do things like get the efficiency of an ntuple dict, bin values and
take a measure on each set of binned values, and create custom value
lists that wouldn't be found in the original ntuple.
"""

from . import operations as ndops


def get_proportion_selected(val_list, selector, norm=True):
    """Find the proportion of tracks selected with the given selector.
    If there are no tracks in the tracks property value list, returns
    zero. Can also return the number of tracks meeting the condition.

    Args:
        val_list: a list of values of a track property, such as
            tp_pt or trk_chi2rphi.
        selector: a property that these value can satisfy. For
            example, "lambda trk_eta: trk_eta <= 2.4".
        norm: if True, divides the number of tracks meeting the
            condition by the total number of tracks. This is the default
            option.

    Returns:
        Either the number or proportion of tracks meeting the condition,
        depending on the value of norm.
    """

    if len(val_list) == 0:
        return 0

    num_tracks_meeting_cond = sum(map(selector, val_list))
    return float(num_tracks_meeting_cond) / len(val_list) if norm \
            else num_tracks_meeting_cond


def make_bins(bin_specifier, binning_values):
    """Takes in a bin specifier, which is either an integer number of
    bins, a tuple of the form (lower_bound, upper_bound, num_bins) or
    a list of values, with the last element being the upper bound of the
    last bin.

    If bin_specifier is an integer, it uses the max and min values of
    binned_property to find its range.

    If bin_specifier is a 3-tuple, it creates the third argument number
    of evenly spaced bins between the first two values.

    If bin_specifier is a list, return the list.

    Args:
        bin_specifier: either an int for the number of bins, a 3-tuple
            of the form (low_bound, high_bound, num_bins), or a list of
            numbers
        binning_values: a list of values forming the basis for the bins

    Returns:
        A list of bin edges, of length one greater than the number of
        bins.

    Raises:
        ValueError if bin_specifier is not an int, tuple, or list
    """

    if isinstance(bin_specifier, int):
        bin_specifier = (min(binning_values), max(binning_values),
                bin_specifier)
    if isinstance(bin_specifier, tuple):
        bin_specifier = list(bin_specifier)
        bin_specifier[2] += 1  # we'll need one more value than we want bins
        bin_specifier = list(linspace(*bin_specifier))
    if isinstance(bin_specifier, list):
        return bin_specifier

    raise ValueError("Expected int, tuple, or list as arg 'bin_specifier', "
            "but received {}.".format(str(type(bin_specifier))))


def take_measure_by_bin(track_prop_dict, bin_property, measure, bins=30):
    """Bin a track properties dict by a value list of a corresponding
    property, then compute some measure for the values in each bin. For
    example, the track_prop_dict could could be of tracking particles
    and contain nmatch, and the measure could be
    eff_from_track_prop_dict.

    Args:
        track_prop_dict: a track properties dict.
        bin_property: a property in track_prop_dict that will split it
            into bins. Preferably a continuous value, but no hard
            restriction is made in this code.
        measure: a function that takes in a track properties dict and
            returns a number.
        bins: either an int for the number of bins, a 3-tuple of the
            form (low_bound, high_bound, num_bins), or a list of
            numbers. See ntupledict.operations.make_bins() for info.

    Returns:
        The bins and the bin heights computed from the binned value
        lists.
    """

    binning_val_list = track_prop_dict[bin_property]
    bins = make_bins(bins, binning_val_list)

    # Sort values into bins with respect to binning value
    bin_heights = list(map(lambda lower_bin, upper_bin:
        measure(ndops.cut_track_prop_dict(track_prop_dict,
            # Select values in range lower_bin to upper_bin, but exclude values
            # equal to upper_bin
            {bin_property: lambda val: select(lower_bin, upper_bin)(val) and
                select([select(upper_bin)], invert=True)})),
        bins[:-1], bins[1:]))

    return bins, bin_heights


def eff_from_ntuple_dict(ntuple_dict, tp_selector_dict={}):
    """Finds the efficieny of an ntuple dict. Restrictions can be made
    on the tracking particles by performing a cut on the ntuple. Note
    that the ntuple must contain pt.

    Args:
        ntuple_dict: an ntuple dictionary containing a tracking
            particle track property dict.
        tp_selector_dict: a dictionary from tp properties
            ("pt", "eta", etc.) to conditions (lambda pt: pt < 2, etc.).

    Returns:
        The efficiency of the tracking algorithm for the tracks in the
        given ntuple dict.
    """

    return eff_from_track_prop_dict(ntuple_dict["tp"], tp_selector_dict)


def eff_from_track_prop_dict(track_prop_dict_tp, selector_dict={}):
    """Finds the efficieny of an track properties dict. Restrictions
    can be made on the tracking particles by performing a cut. Note
    that the track properties dictionary must be of tracking particles.

    Args:
        track_prop_dict_tp: a tracks properties dict carrying value
            lists from tracking particles.
        selector_dict: a dictionary from tp properties
            ("pt", "eta", etc.) to conditions (lambda pt: pt < 2, etc.).

    Returns:
        The efficiency of the tracking algorithm run on the given track
        properties dict
    """

    return get_proportion_selected(
            ndops.cut_track_prop_dict(
                track_prop_dict_tp, selector_dict)["nmatch"],
            select(1, inf))


class StubInfo(object):
    """Converts eta and hitpattern into data about stubs. Accessible
    data includes expected number of stubs, missing stubs, number of
    PS and 2S modules expected, hit, missed, etc.

    Note that these definitions are in accordance with the expected and
    missed definitions in the TrackTrigger's Kalman filter used to
    originally create hitpattern. One consequence of this is that there
    will never be hit stub that was not expected."""

    def __init__(self, eta, hitpattern):
        """Stores expected, hit, and PS (False for 2S) as tuples of
        boolean values. Indices 0 - 5 in the lists correspond to layers
        1 - 6, while indices 6 - 10 in the list correspond to disks
        1 - 5."""

        self._gen_expected(abs(eta))
        self._gen_hit(hitpattern)
        self._ps = (True, True, True, False, False, False,  # layers
                    True, True, False, False, False)        # disks

    def _gen_expected(self, abseta):
        """Sets a tuple of boolean values indicating whether the
        Kalman filter expects a hit on a layer/disk for some absolute
        eta. If eta is greater than 2.4, the list will be all False."""

        # eta regions and the indices of expected layers/disks
        eta_regions = [0., 0.2, 0.41, 0.62, 0.9, 1.26, 1.68, 2.08, 2.4]
        num_layers_disks = 11
        layer_maps = [[1,  2,  3,  4,  5,  6],
                      [1,  2,  3,  4,  5,  6],
                      [1,  2,  3,  4,  5,  6],
                      [1,  2,  3,  4,  5,  6],
                      [1,  2,  3,  4,  7,  8,  9],
                      [1,  2,  3,  7,  8,  9, 10],
                      [1,  2,  8,  9, 10, 11],
                      [1,  7,  8,  9, 10, 11]]

        expected_layers = []
        for eta_low, eta_high, layer_map in zip(
                eta_regions[:-1],
                eta_regions[1:],
                layer_maps):
            if eta_low <= abseta <= eta_high:
                expected_layers = layer_map
                break

        self._expected = tuple(map(lambda index: index + 1 in expected_layers,
            range(num_layers_disks)))

    def _gen_hit(self, hitpattern):
        """Generates a tuple of the same form as the expected hits tuple
        using the hitpattern variable and the expected hits list. Each
        True value in this list represents a hit. The _gen_expected()
        method must be run first."""

        def gen_hits_iter(hitpattern, num_expected):
            """Return an iterator through hitpattern by converting it
            into a list of boolean values, ordered by ascending
            magnitude in the original hitpattern. Falses are included
            at the end of the list until it is the same length as the
            expected number of values (6 or 7)."""

            hits_bool = [bool(int(i)) for i in bin(hitpattern)[-1:1:-1]]
            return iter(hits_bool + (num_expected - len(hits_bool)) * [False])

        hits_iter = gen_hits_iter(hitpattern, len(self._expected))
        self._hit = tuple(map(lambda expected:
            expected and next(hits_iter),
            self._expected))


def create_stub_info_list(track_prop_dict):
    """Using eta and hitpattern, draws info about which layers in the
    outer tracker were and weren't hit from a track properties dict.

    Args:
        track_prop_dict: a tracks properties dict with track properties
            eta and hitpattern. Must represent either trk or matchtrk,
            as only those have the hitpattern track property.

    Returns:
        A list of StubInfo objects, one for each track, from which all
        stub data can be derived.
    """

    return list(map(lambda eta, hitpattern:
        StubInfo(eta, hitpattern),
        track_prop_dict["eta"], track_prop_dict["hitpattern"]))

