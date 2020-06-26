"""Analyzes the contents of ntuple dicts, track property dicts, and
value lists.

Do things like get the efficiency of an ntuple dict, bin values and
take a measure on each set of binned values, and create custom value
lists that wouldn't be found in the original ntuple.

Also contains a function for finding the error of a prediction given
the prediction and the real values. This can be a prediction as per
efficiency of track finding or a prediction by an ML model.
"""

from . import operations as ndops
from .operations import select as sel
from numpy import linspace
from math import sqrt
from statistics import stdev


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
            "but received {}.".format(str(bin_specifier)))


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
            returns a number and an error.
        bins: either an int for the number of bins, a 3-tuple of the
            form (low_bound, high_bound, num_bins), or a list of
            numbers. See ntupledict.operations.make_bins() for info.

    Returns:
        The bins, bin heights, and errors computed from the binned value
        lists.
    """

    binning_val_list = track_prop_dict[bin_property]
    bins = make_bins(bins, binning_val_list)

    # Sort values into bins with respect to binning value
    bin_heights_and_errs = list(map(lambda lower_bin, upper_bin:
        measure(ndops.cut_track_prop_dict(track_prop_dict,
            # Select values in range lower_bin to upper_bin,
            # but exclude values equal to upper_bin
            {bin_property: lambda val: lower_bin <= val < upper_bin})),
        bins[:-1], bins[1:]))

    bin_heights = list(map(lambda l: l[0], bin_heights_and_errs))
    bin_errs = list(map(lambda l: l[1], bin_heights_and_errs))

    return bins, bin_heights, bin_errs


def pred_error(domain_size, num_selected):
    """Finds the error of a prediction in some domain given the size of
    the domain and the number of correct predictions in that domain. If
    at any point division by zero is attempted, return 0."""

    try:
        return 1 / (domain_size * sqrt(
            num_selected * (1 - (num_selected / domain_size))))
    except ZeroDivisionError:
        return 0


def eff_from_ntuple_dict(ntuple_dict, tp_selector_dict=None):
    """Finds the efficieny of an ntuple dict and its standard deviation.
    Restrictions can be made on the tracking particles by performing a
    cut on the ntuple. Note that the ntuple must contain pt.

    Args:
        ntuple_dict: an ntuple dictionary containing a tracking
            particle track property dict.
        tp_selector_dict: a dictionary from tp properties
            ("pt", "eta", etc.) to conditions (lambda pt: pt < 2, etc.).

    Returns:
        A tuple containing the efficiency of the tracking algorithm for
        the tracks in the given ntuple dict and the standard deviation.
    """

    return eff_from_track_prop_dict(ntuple_dict["tp"], tp_selector_dict)


def eff_from_track_prop_dict(track_prop_dict_tp, selector_dict=None):
    """Finds the efficieny with pred error of an track properties dict.
    Restrictions can be made on the tracking particles by performing a
    cut. Note that the track properties dictionary must be of tracking
    particles.

    Args:
        track_prop_dict_tp: a tracks properties dict carrying value
            lists from tracking particles.
        selector_dict: a dictionary from tp properties
            ("pt", "eta", etc.) to conditions (lambda pt: pt < 2, etc.).

    Returns:
        A tuple containing the efficiency of the tracking algorithm for
        the tracks in the given ntuple dict and the standard deviation.
    """

    if selector_dict is not None:
        track_prop_dict_tps = ndops.cut_track_prop_dict(
                track_prop_dict_tp, selector_dict)

    num_tps = ndops.track_prop_dict_length(track_prop_dict_tps)
    num_matched_tps = num_tps - track_prop_dict_tps["nmatch"].count(0)

    return num_matched_tps / num_tps, pred_error(num_tps, num_matched_tps)


class StubInfo(object):
    """Converts eta and hitpattern into data about stubs for a single
    track.

    The only directly accessible info from this class are boolean
    lists, all of which are indexed by layer/disk:
        - indices 0 - 5 in the lists correspond to layers 1 - 6.
        - indices 6 - 10 in the list correspond to disks 1 - 5.
    Any information you could want about stubs can be found from these
    three lists, sum(), map(), and lambda.

    For example, if I wanted to find the number of missing 2S layers:

        def missing_2S_layers(stub_info):
            return sum(map(lambda expected, hit, ps_2s:
                            not ps_layer and expected and not hit,
                            stub_info.get_expected(),
                            stub_info.get_hit(),
                            stub_info.get_ps_2s()))

    Down below, there are convenience functions process_stub_info and
    basic_process_stub_info for processing instances of this class.

    Note that these definitions are in accordance with the expected and
    missed definitions in the TrackTrigger's Kalman filter used to
    originally create hitpattern. One consequence of this is that there
    will never be hit stub that was not expected.
    """

    def __init__(self, eta, hitpattern):
        """Stores expected, hit, and PS (False for 2S) as tuples of
        boolean values."""

        self._gen_expected(abs(eta))
        self._gen_hit(hitpattern)
        self._gen_ps_2s(abs(eta))

    def _gen_expected(self, abseta):
        """Sets a tuple of boolean values indicating whether the
        Kalman filter expects a hit on a layer/disk for some absolute
        eta. If eta is greater than 2.4, the list will be all False.

        Args:
            abseta: the absolute value of a pseudorapitiy measurement
        """

        # eta regions for and indices of expected layers/disks
        eta_regions = [0., 0.2, 0.41, 0.62, 0.9, 1.26, 1.68, 2.08, 2.4]
        num_layers_disks = 11
        layer_maps = [[1,  2,  3,  4,  5,  6],
                      [1,  2,  3,  4,  5,  6],
                      [1,  2,  3,  4,  5,  6],
                      [1,  2,  3,  4,  5,  6],
                      [1,  2,  3,  4,  5,  6],
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
        method must be run first.

        Args:
            hitpattern: a number that, when in base two, corresponds to
                a list of zeroes or ones that indicate whether each
                layer in a set of six or seven expected layers were hit.
        """

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
            self.get_expected()))

    def _gen_ps_2s(self, abseta):
        """Generates a tuple indexed by layer for which each boolean
        value represents whether a layer or disk is PS (True) or 2S
        (False). This is necessary because a given disk has PS and 2S
        modules, separated by eta.

        Args:
            abseta: the absolute value of a pseudorapitiy measurement
        """

        layer_ps_2s = 3 * (True,) + 3 * (False,)

        disk_ps_2s_cuts = [1.45, 1.6, 1.8, 1.975, 2.15]
            # ps above, 2s below
        disk_ps_2s = tuple(map(lambda disk_ps_2s_cut:
            abseta > disk_ps_2s_cut,
            disk_ps_2s_cuts))

        self._ps_2s = layer_ps_2s + disk_ps_2s

    def get_expected(self):
        """Returns a list of booleans representing which layers/disks
        were expected to be hit by the Kalman filter."""

        return list(self._expected)

    def get_hit(self):
        """Returns a list of booleans representing which layers/disks
        were hit, within the layers/disks expected byt the Kalman
        filter."""

        return list(self._hit)

    def get_ps_2s(self):
        """Returns a list of booleans indexed by layer/disk indicating
        if the layer or disk with that index is PS (True) or 2S
        (False)."""

        return list(self._ps_2s)


def create_stub_info_list(track_prop_dict, process_stub_info):
    """Uses eta and hitpattern to generate a list of StubInfos from the
    given track property dict. Then maps those StubInfos to something
    else using some function.

    Args:
        track_prop_dict: a tracks properties dict with track properties
            eta and hitpattern. Must represent either trk or matchtrk,
            as only those have the hitpattern track property.
        process_stub_info: a function or lambda expression that accepts
            StubInfos.

    Returns:
        A list of processed StubInfos indexed by track.
    """

    return list(map(lambda eta, hitpattern:
        process_stub_info(StubInfo(eta, hitpattern)),
        track_prop_dict["eta"], track_prop_dict["hitpattern"]))


def basic_process_stub_info(process_layer):
    """Returns a StubInfo processing function that is agnostic towards
    layer indices, which means it should work for most cases.

    For example, a function that determines how many missing 2S layers
    are in a StubInfo would be:

        basic_process_stub_info(lambda expected, hit, ps_2s:
                                not ps_2s and expected and not hit)

    Args:
        process_layer: A function from a single layer's expected bool,
        hit bool, and ps/2s bool (in that order) to a boolean.

    Returns:
        A function that accepts a StubInfo and counts for how many
        layers process_layer returns True.
    """

    return lambda stub_info: sum(map(process_layer,
                                     stub_info.get_expected(),
                                     stub_info.get_hit(),
                                     stub_info.get_ps_2s()))

