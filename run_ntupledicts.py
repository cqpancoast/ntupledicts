from sys import argv
from uproot import open as uproot_open
from ntupledicts import ntuple_dict_operations as ndops
from ntupledicts.ntuple_dict_operations import select as sel
from ntupledicts import ntuple_dict_plot as ndplot


"""
UPROOT NTUPLE PLOT

An uproot-based child of the TrackFindingTracklet package's L1TrackNtuplePlot.
It is friendly with machine learning aims and goals, as our ML already takes
place in Python.

There are three parts to this code:
    - Getting Data.
        One or more ntuples is read in. If more than one ntuple is
        read in, they are all concatenated together. More on the data
        structures used below.
    - Applying Cuts.
        Place arbitrary cuts on the event collection on any property of
        any track type.
    - Plotting.
        The matplotlib library is (for some) easier to work with than
        the ROOT plotting library. However, matplotlib does not know you
        are a pysicist! This is not intended as a replacement for ROOT,
        just an alternative.


Data Definitions and Scheme

All event data is stored in an object called an ntuple dictionary. This is a
dictionary from track types ("trk", "tp", "matchtrk", "matchtp") to dicts
from track properties ("eta", "chi2", "nmatch") to lists of properties. (These
smaller dicts within the ntuple dictionaries are called "track property dicts".)
For example, a simple ntuple dictionary might look like this:

    {"trk": {"pt": [1, 2, 3], "eta": [0, 2.2, 1.1]}, "tp": {"nmatch": ...}}

The values here are the track property dictionaries.

In the main method, you can specify which properties you want to access from
the ntuple to have in your track type dictionary. This is nice, because you don't
have to deal with hauling around any more data than you have to!


Applying Cuts

To perform a cut on a track type dictionary... TODO docs!!!


Plotting

All plotting functions follow certain rules:
    - All return an axes object
    - None clear or save the figure
    - Some set labels (x-axis, y-axis, title) but some do not


TODO (in mixed order of dependence/importance):
- standardize language in code to match data def scheme in documentation
    - after that, complete documentation
- make it play nice with ML stuff
    - Include Claire's plotting functions (or altered versions of them) in
        plot file
- reduce import overhead; this is ridiculous
    - maybe it's only ridiculous for me on lxplus though
- reduce every line down to 80 characters to comply with Python standards
- unit tests? Is that the physicist way? (Do I have the time or patience?)
- fix ROC curve plot fxn whenever you feel like it
    - doing this requires making selectors into objects so that you can
        get info about them explicitly

"""


# Global variables
# (It isn't good practice, but this is certainly better than passing them around like dead weight.)

# Input/Output global variable dict
input_files = ["data/eventsets/TTbar_PU200_D49_20.root"]
input_file_short = "ttbar_pu200_20"
output_dir = "savedplots/"


def main(argv):

    # Open ntuples, specify desired properties and cuts to be applied
    event_sets = []
    for input_file in input_files:
        event_sets.append(next(iter(uproot_open(input_file).values()))["eventTree"])
    properties_by_track_type = {"trk": ["pt", "eta", "nstub", "chi2rphi", "chi2rz", "genuine", "fake"],
                                "matchtrk": ["pt", "eta", "nstub", "chi2rphi", "chi2rz"],
                                "tp": ["pt", "eta", "nstub", "dxy", "d0", "eventid", "nmatch"]}
    cut_dicts = {"tp": {"eta": sel(-2.4, 2.4), "pt": sel(2, 100), "nstub": sel(4, 999),
                        "dxy": sel(-1.0, 1.0), "d0": sel(-1.0, 1.0), "eventid": sel(0)}}

    # Create ntuple properties dict from event set
    ntuple_properties = ndops.ntuples_to_ntuple_dict(event_sets, properties_by_track_type)
    ntuple_properties = ndops.cut_ntuple(ntuple_properties, cut_dicts)

    go(ntuple_properties)

    print("Process complete. Exiting program.")


def go(ntuple_properties):
    """Call functions from the file where all the plots are!

    Args:
        ntuple_properties:  you know, the thing
    """

    # Call appropriate plotting function(s)

    # TODO machine learning stuff here

    """
    ax = ndplot.plot_roc_curve(ntuple_properties, "chi2rphi", [sel(0, 9999), sel(0, 500), sel(0, 100), sel(0, 70), sel(0, 50), sel(0, 40), sel(0, 30), sel(0, 20), sel(0, 10), sel(0, 5)], group_name="allpt")
    #axz = ndplot.plot_roc_curve(ntuple_properties, "chi2rz", [sel(0, 9999), sel(0, 500), sel(0, 50), sel(0, 20), sel(0, 11), sel(0, 10), sel(0, 9), sel(0, 8), sel(0, 7), sel(0, 6), sel(0, 5)], group_name="allpt")
    ntuple_properties = ndops.cut_ntuple(ntuple_properties, {"trk": {"pt": sel(8, 100)}, "tp": {"pt": sel(8, 100)}})
    ndplot.plot_roc_curve(ntuple_properties, "chi2rphi", [sel(0, 9999), sel(0, 500), sel(0, 100), sel(0, 70), sel(0, 50), sel(0, 40), sel(0, 30), sel(0, 20), sel(0, 10), sel(0, 5)], group_name="pt8", ax=ax, color='orange')
    ndplot.plt.savefig("beepbooptest.pdf")
    #ndplot.plot_roc_curve(ntuple_properties, "chi2rz", [sel(0, 9999), sel(0, 500), sel(0, 50), sel(0, 20), sel(0, 11), sel(0, 10), sel(0, 9), sel(0, 8), sel(0, 7), sel(0, 6), sel(0, 5)], group_name="pt8")#, ax=axz, color='orange')


    genuine_properties = ndops.cut_trackset(
        ntuple_properties["trk"], {"genuine": sel(0)})
    not_genuine_properties = ndops.cut_trackset(
        ntuple_properties["trk"], {"genuine": sel(1)})
    fake_properties = ndops.cut_trackset(
        ntuple_properties["trk"], {"fake": sel(0)})
    not_fake_properties_primary = ndops.cut_trackset(
        ntuple_properties["trk"], {"fake": sel(1)})
    not_fake_properties_secondary = ndops.cut_trackset(
        ntuple_properties["trk"], {"fake": sel(2)})

    ndplot.overlay({"primary-vertex": not_fake_properties_primary["chi2rphi"],
             "secondary-vertex": not_fake_properties_secondary["chi2rphi"],
             "fake": fake_properties["chi2rphi"]}, "chi2rphi", "trk_fake_dist")
    ndplot.overlay({"primary-vertex": not_fake_properties_primary["chi2rz"],
             "secondary-vertex": not_fake_properties_secondary["chi2rz"],
             "fake": fake_properties["chi2rz"]}, "chi2rz", "trk_fake_dist")
    ndplot.overlay({"genuine": genuine_properties["chi2rphi"],
             "fake": not_genuine_properties["chi2rphi"]}, "chi2rphi", "trk_genuine_dist")
    ndplot.overlay({"genuine": genuine_properties["chi2rz"],
             "fake": not_genuine_properties["chi2rz"]}, "chi2rz", "trk_genuine_dist")
    """


main(argv)
