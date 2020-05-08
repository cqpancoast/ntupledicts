from numpy import linspace
from copy import deepcopy
from math import sqrt
import matplotlib as mpl
import matplotlib.pyplot as plt
import operations as ndops


def plot_roc_curve_from_cut_list(ntuple_dict_in,
        cut_property, cuts, cuts_increasing=True,
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
        cuts_increasing: if the cuts are strictly in increasing order of 
            strictness, the same list can be preserved, decreasing runtime.
            True by default
        group_name: if you are plotting this data with other data, this
            will be the legend label
        ax: if you are plotting this data with other data, you can pass
            a previous axis in
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
                if cuts_increasing else ntuple_dict["trk"]
        trackset_to_cut["matchtrk"] = cut_matchtracks_properties\
                if cuts_increasing else ntuple_dict["matchtrk"]
        cut_tracks_properties = ndops.cut_trackset(
            trackset_to_cut["trk"], {cut_property: ndops.select(*cut)})
        cut_matchtracks_properties = ndops.cut_trackset(
            trackset_to_cut["matchtrk"], {cut_property: ndops.slect(*cut)})
        cuts_plot_info["cut"].append(cut[1])  # only get upper bound of cut
        cuts_plot_info["eff"].append(eff_from_ntuple(
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


#TODO this function is not just deprecated; it's CONDEMNED.
def proportion_foreach_bin(trks_property, trks_property_wrt, property_condition,
        num_bins=30, ax=plt.figure().add_subplot(111)):
    """wrt meaning "with respect to" - bin tracks by property, and then see how another
    property of those tracks meets a certain condition. For example, fake rate by eta,
    or efficiency by pT.

    Bear in mind that this function only has stdev error bars. One creating a graph like
    this should generally use ROOT.
    """

    bins = linspace(min(trks_property_wrt), max(trks_property_wrt), num_bins)

    # Sort tracks into bins
    bins_trks_property = [[] for bin in bins[:-1]]
    for trk_property, trk_property_wrt in zip(trks_property, trks_property_wrt):
        # reversed iteration makes checking values against lower bin edges easier
        for bindex, bin in zip(reversed(range(len(bins))), reversed(bins)):
            if trk_property_wrt > bin:
                bins_trks_property[bindex].append(trk_property)
                break

    # For each bin, find tracks meeting condition, get ratio and stdev
    bins_trks_proportions = []
    bins_trks_proportions_stdev = []
    for bin_trks_property in bins_trks_property:
        bins_trks_proportions.append(get_proportion_meeting_condition(
            bin_trks_property, property_condition))
        bins_trks_proportions_stdev.append(
            1/sqrt(len(bin_trks_property)) if len(bin_trks_property) > 0 else 0)

    ax.errorbar(x=[(bin_low_edge + bin_high_edge) / 2 for bin_low_edge, bin_high_edge in zip(bins[:-1], bins[1:])],
                 y=bins_trks_proportions, yerr=bins_trks_proportions_stdev,
                 elinewidth=1, drawstyle="steps-mid")

    return ax


def hist2D_properties(property1, property2, property1name="", property2name="",
        group_name="", bins=30):
    """Make a 2D histogram with property 1 on the x axis and property 2 on 
    the y axis."""

    cmap = mpl.cm.get_cmap("viridis")
    cmap.set_under("w")

    ax = plt.figure().add_subplot(111)
    ax.hist2d(property1, property2, bins=bins,
               norm=mpl.colors.LogNorm(vmin=.5))
    ax.colorbar()
    ax.set_xlabel(property1name)
    ax.set_ylabel(property2name)
    ax.set_title(property1 + " vs. " + property2 + " for " + group_name + " tracks")

    return ax

