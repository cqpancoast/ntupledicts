from numpy import linspace
from copy import deepcopy
from math import sqrt
import matplotlib as mpl
import matplotlib.pyplot as plt
from ntupledicts import *


def get_proportion_meeting_condition(tracks_property, property_condition, norm=True):
    """Find the proportion of tracks whose given property meets a condition.
    If the number of tracks is zero, returns zero. Can also return the number
    of tracks meeting the condition.

    Args:
            tracks_property:  a list of values of a track property, such as
                    trk_pt or tp_chi2rphi
            property_condition:  a property that these value can satisfy. For
                    exampe, "lambda trk_eta: trk_eta <= 2.4".
            norm:  if True, divides the number of tracks meeting the condition
                    by the total number of tracks. This is the default option.

    Returns:
            Either the number or proportion of tracks meeting the condition,
            depending on the value of norm.
    """

    if len(tracks_property) == 0:
        print("Cannot calculate proportion meeting condition in zero-length quantity. Returning zero.")
        return 0

    num_tracks_meeting_cond = sum(map(property_condition, tracks_property))
    return float(num_tracks_meeting_cond) / len(tracks_property) if norm else num_tracks_meeting_cond


def eff_from_ntuple(ntuple_properties, tp_cond_dict={}):
    """Finds the efficieny of an ntuple. Restrictions can be made on the 
    tracking particles by performing a cut on the ntuple. Note that the
    ntuple must contain pt.

    Args:
            ntuple_properties:  the ntuple to find the efficiency of
            tp_cond_dict:  a dictionary from tp properties ("pt", "eta", etc.)
                    to conditions (lambda pt: pt < 2, etc.)

    Returns:
            The efficiency of the tracking algorithm run on the given ntuple
    """

    # Cutting on tracking particles also cuts the corresponding matchtracks
    cut_ntuple_properties = cut_ntuple(ntuple_properties, {"tp": tp_cond_dict})
    tps_nmatch = cut_ntuple_properties["tp"]["nmatch"]

    # Now, count how many tp's have an nmatch value greater than zero
    return float(sum(map(lambda nmatch: nmatch > 0, tps_nmatch))) / len(tps_nmatch)


def plot_roc_curve(ntuple_properties_in, cut_property, cuts, cuts_increasing=True, group_name="",
                   ax=plt.figure().add_subplot(111), save_plot=True, color='blue'):
    """Adjusts the cut on a certain variable in an event set and plots the change in
    tracking efficiency and fake rate.

    Args:
            ntuple_properties_in:  a dict from track types to dicts from track properties
                    to lists of values. The input value is unaltered by this function
            cut_property:  the variable to change cuts on
            cuts:  a list of length 2 lists containing lower and upper cut bounds (inclusive)
            cuts_increasing:  if the cuts are strictly in increasing order of strictness, the
                    same list can be preserved, decreasing runtime. True by default
            group_name: if you are plotting this data with other data, this will be the
                    legend label
            ax: if you are plotting this data with other data, you can pass a previous axis in
            save_plot: whether to save the plot

    Returns:
            The axes object used to plot this graph, for use in overlaying
    """

    # We don't want to alter our original ntuple_properties
    ntuple_properties = deepcopy(ntuple_properties_in)

    # Build up cuts plot info, which is what will be plotted
    cuts_plot_info = {"cut": [], "eff": [], "fake_rate": []}
    trackset_to_cut = {}
    cut_tracks_properties = ntuple_properties["trk"]
    cut_matchtracks_properties = ntuple_properties["matchtrk"]
    for cut in cuts:
        trackset_to_cut["trk"] = cut_tracks_properties if cuts_increasing else ntuple_properties["trk"]
        trackset_to_cut["matchtrk"] = cut_matchtracks_properties if cuts_increasing else ntuple_properties["matchtrk"]
        cut_tracks_properties = cut_trackset(
            trackset_to_cut["trk"], {cut_property: cut})
        cut_matchtracks_properties = cut_trackset(
            trackset_to_cut["matchtrk"], {cut_property: cut})
        cuts_plot_info["cut"].append(cut[1])  # only get upper bound of cut
        cuts_plot_info["eff"].append(eff_from_matchtrks_and_tps(
            cut_matchtracks_properties["pt"], ntuple_properties["tp"]["pt"]))
        cuts_plot_info["fake_rate"].append(get_proportion_meeting_condition(
            cut_tracks_properties["genuine"], lambda gen_track: gen_track == 0))

    # Now plot!
    ax.plot(cuts_plot_info["fake_rate"], cuts_plot_info["eff"],
            "b.", label=group_name, color=color)

    if save_plot:
        ax.set_xlabel("Track fake rate")
        ax.set_ylabel("Tracking efficiency")
        ax.legend()
        # for fake_rate, eff, cut in zip(cuts_plot_info["fake_rate"], cuts_plot_info["eff"], cuts_plot_info["cut"]):
        #ax.annotate(str(cut), ax=(fake_rate, eff))
        ax.set_title("Fake rate and efficiency for changing " +
                     cut_property + " cuts " + group_name)

    return ax


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


def overlay(trackset_dict, overlaid_property="", group_name="", num_bins=30):
    """Takes in a dictionary where each value is a list of values of overlaid_property
    and each key identifies what makes its value set unique. For example, the dict
    could map from the string "fake" to the list of overlaid_properties values from
    fake tracks, and does the same for real tracks."""

    trackset_properties = iter(trackset_dict.keys())
    trackset_properties_values = iter(trackset_dict.values())

    ax = plt.figure().add_subplot(111)
    _, bins, _ = ax.hist(next(trackset_properties_values), bins=num_bins, log=True, 
            density=True, histtype="step", label=next(trackset_properties))
    
    trackset_property = next(trackset_properties, None)
    trackset_property_values = next(trackset_properties_values, None)
    while trackset_property is not None and trackset_property_values is not None:
        ax.hist(trackset_property_values, bins=bins, log=True, density=True,
                 histtype="step", label=trackset_property)
        trackset_property = next(trackset_properties, None)
        trackset_property_values = next(trackset_properties_values, None)
       
    ax.set_xlabel(overlaid_property)
    ax.set_ylabel("# of tracks")
    ax.legend()
    ax.set_title(r"Normalized " + overlaid_property + " values for tracks," + group_name)
    
    return ax


def hist2D_properties(property1, property2, property1name="", property2name="",
        group_name="", bins=30):
    """Make a 2D histogram with property 1 on the x axis and property 2 on 
    the y axis."""

    cmap = mpl.cm.viridis
    cmap.set_under("w")

    ax = plt.figure().add_subplot(111)
    ax.hist2d(property1, property2, bins=bins,
               norm=mpl.colors.LogNorm(vmin=.5))
    ax.colorbar()
    ax.set_xlabel(property1name)
    ax.set_ylabel(property2name)
    ax.set_title(property1 + " vs. " + property2 + " for " + group_name + " tracks")

    return ax

