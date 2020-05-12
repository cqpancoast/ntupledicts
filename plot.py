from copy import deepcopy
import matplotlib as mpl
import matplotlib.pyplot as plt
from . import operations as ndops
from .operations import select as sel


#FIXME complicated refactoring required
def plot_roc_curve_from_cut_list(ntuple_dict_in,
        cut_property, cuts, cuts_constricting=True,
        group_name="", ax=plt.figure().add_subplot(111), color='blue'):
    """Adjusts the cut on a certain variable in an event set and plots
    the change in tracking efficiency and fake rate.

    Args:
        ntuple_dict_in: a dict from track types to dicts from track 
            properties to lists of values. The input value is unaltered
            by this function
        cut_property: the variable to change cuts on
        cuts: a list of length 2 lists containing lower and upper cut 
            bounds (inclusive). These are used rather than selectors so
            that information about the cut can be used in the final graph
        cuts_constricting: if the cuts are strictly in increasing order of 
            strictness, the same list can be preserved, decreasing runtime.
            True by default
        group_name: if you are plotting this data with other data, this
            will be the legend label
        ax: an axes object to overlay this data onto a previous plot
        color: the color to plot the curve in

    Returns:
        The axes object used to plot this graph, for use in overlaying
    """

    # We don't want to alter our original ntuple_dict
    ntuple_dict = deepcopy(ntuple_dict_in)

    # Build up cuts plot info, which is what will be plotted
    cuts_plot_info = {"cut": [], "eff": [], "fake_rate": []}
    trackset_to_cut = {}
    cut_ntuple_dict = ntuple_dict
    for cut in cuts:
        trackset_to_cut["trk"] = cut_tracks_properties\
                if cuts_constricting else ntuple_dict["trk"]
        trackset_to_cut["matchtrk"] = cut_matchtracks_properties\
                if cuts_constricting else ntuple_dict["matchtrk"]
        cut_tracks_properties = ndops.cut_trackset(
            trackset_to_cut["trk"], {cut_property: sel(*cut)})
        cut_matchtracks_properties = ndops.cut_trackset(
            trackset_to_cut["matchtrk"], {cut_property: sel(*cut)})
        cuts_plot_info["cut"].append(cut[1])  # only get upper bound of cut
        cuts_plot_info["eff"].append(ndops.eff_from_ntuple_dict(
            cut_matchtracks_properties["pt"], ntuple_dict["tp"]["pt"]))
        cuts_plot_info["fake_rate"].append(ndops.get_proportion_selected(
            cut_tracks_properties["genuine"], lambda gen_track: gen_track == 0))

    # Now plot!
    ax.plot(cuts_plot_info["fake_rate"], cuts_plot_info["eff"],
            "b.", label=group_name, color=color)

    ax.set_xlabel("Track fake rate")
    ax.set_ylabel("Tracking efficiency")
    ax.legend()
    for fake_rate, eff, cut in zip(
            cuts_plot_info["fake_rate"],
            cuts_plot_info["eff"],
            cuts_plot_info["cut"]):
        ax.annotate(str(cut), ax=(fake_rate, eff))
    ax.set_title("Fake rate and efficiency for changing " +
                 cut_property + " cuts " + group_name)

    return ax


def plot_measure_by_bin(track_prop_dict, bin_property, measure,
        num_bins=30, ax=plt.figure().add_subplot(111)):
    """Bin a track properties dict by a value list of a corresponding
    property, compute some measure of the values in each bin, then plot.

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

