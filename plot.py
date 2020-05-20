from copy import deepcopy
import matplotlib as mpl
import matplotlib.pyplot as plt
from . import operations as ndops
from .operations import select as sel


def plot_roc_curve_from_cut_list(ntuple_dict, cut_property, cuts,
        cuts_constricting=True, ax=plt.figure().add_subplot(111)):
    """Adjusts the cut on a certain variable in an event set and plots
    the change in tracking efficiency and fake rate.

    Args:
        ntuple_dict: an ntuple dict that contains trk with at least
            genuine and the cut property, matchtrk with at least the
            cut property, and tp with at least nmatch
        cut_property: the variable to change cuts on
        cuts: a list of length 2 lists containing lower and upper cut 
            bounds (inclusive). These are used rather than selectors so
            that information about the cut can be used in the final graph
        cuts_constricting: if the cuts are strictly in increasing order of 
            strictness, the same list can be preserved, decreasing runtime.
            True by default
        ax: an axes object to be used to plot this graph

    Returns:
        The axes object used to plot this graph
    """

    # Build up cuts plot info, which is what will be plotted
    cuts_plot_info = {"cut": [], "eff": [], "fake_rate": []}
    tpds_to_cut = {}
    cut_ntuple_dict = ntuple_dict
    for cut in cuts:
        # If cuts are constricting, use cut ntuple dict from previous iteration
        ntuple_dict_to_cut = cut_ntuple_dict if cuts_constricting\
                else ntuple_dict
        tpd_selector = {cut_property: sel(*cut)}
        cut_ntuple_dict = ndops.cut_ntuple(ntuple_dict_to_cut,
                {"trk": tpd_selector, "matchtrk": tpd_selector})
        cuts_plot_info["cut"].append(cut[1])  # only get upper bound of cut
        cuts_plot_info["eff"].append(ndops.eff_from_ntuple_dict(
            cut_ntuple_dict["tp"]["pt"]))
        cuts_plot_info["fake_rate"].append(ndops.get_proportion_selected(
            cut_ntuple_dict["trk"]["genuine"], sel(0)))

    # Now plot!
    ax.plot(cuts_plot_info["fake_rate"], cuts_plot_info["eff"],
            "b.", label=group_name)

    ax.set_xlabel("Track fake rate")
    ax.set_ylabel("Tracking efficiency")
    ax.legend()
    for fake_rate, eff, cut in zip(
            cuts_plot_info["fake_rate"],
            cuts_plot_info["eff"],
            cuts_plot_info["cut"]):
        ax.annotate(str(cut), ax=(fake_rate, eff))

    return ax


def plot_measure_by_bin(track_prop_dict, bin_property, measure,
        bins=30, ax=plt.figure().add_subplot(111)):
    """Splits a track property dict into bins by some value, then
    compute a measure on each binned track property dict.

    Args:
        track_prop_dict: a track properties dict
        bin_property: a property in track_prop_dict that will split it
            into bins. Preferably a continuous value, but no hard
            restriction is made in this code
        measure: a function that takes in a track properties dict and
            returns a number
        bins: either an int for the number of bins, a 3-tuple of the
            form (low_bound, high_bound, num_bins), or a list of
            numbers. See ntupledict.operations.make_bins() for info
        ax: an axes object to overlay this data onto a previous plot

    Returns:
        A matplotlib.pyplot.Axes object for adjusting plot properties
        and overlaying data
    """

    bins, bin_heights = ndops.take_measure_by_bin(track_prop_dict,
            bin_property, measure, bins)
    bin_middles = list(map(lambda lower, upper: (lower + upper) / 2,
        bins[:-1], bins[1:]))

    ax.scatter(bin_middles, bin_heights)
    ax.set_xlabel(bin_property)

    return ax


def plot_property_hist(track_prop_dict, track_property, bins=30,
        ax=plt.figure().add_subplot(111)):
    """Plot a histogram distribution of a track property in a track
    properties dict.

    Args:
        track_prop_dict: a track properties dict
        track_property: a property in track_prop_dict that will split it
            into bins. Preferably a continuous value, but no hard
            restriction is made in this code
        bins: either an int for the number of bins, a 3-tuple of the
            form (low_bound, high_bound, num_bins), or a list of
            numbers. See ntupledict.operations.make_bins() for info
        ax: an axes object to overlay this data onto a previous plot

    Returns:
        A matplotlib.pyplot.Axes object for adjusting plot properties
        and overlaying data
    """

    ax = plot_measure_by_bin(track_prop_dict, track_property,
            lambda tpd: len(tpd[track_property]), bins, ax)
    ax.set_ylabel("num. tracks")
    ax.set_title("Histrogram of {}".format(track_property))

    return ax

