from uproot import open as uproot_open
from sys import argv
from ntupledicts import *


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

All event data is stored in an object called a track type dictionary. This is a
dictionary from track types ("trk", "tp", "matchtrk", "matchtp") to dicts
from track properties ("eta", "chi2", "nmatch") to lists of properties. (These
smaller dicts within the track type dictionaries are called "track property dicts".)
For example, a simple track type dictionary might look like this:

	{"trk": {"pt": [1, 2, 3], "eta": [0, 2.2, 1.1]}, "tp": {"nmatch": ...}}

The values here are the track property dictionaries.

In the main method, you can specify which properties you want to access from
the ntuple to have in your track type dictionary. This is nice, because you don't
have to deal with hauling around any more data than you have to!


Applying Cuts

To perform a cut on a track type dictionary... TODO docs!!!


Plotting

The plotting functions contained here are, so far, a big mess of things that
were useful to me at one point. If I ship this out to people, I want to have a
clearer organization scheme with these things.

Starting with something like, all functions return axes objects?


TODO:
- make dealing with multiple event sets possible
    - in addition, have operations for event set concatenation
- standardize language in code to match data def scheme in documentation
- after that, complete documentation
- change cuts from crappy 2-arrays to configurable function objects
- reduce import overhead; this is ridiculous
- reduce every line down to 80 characters to comply with Python standards
- unit tests? Is that the physicist way? (Do I have the time or patience?)
- "all plotting functions return axes objects"
- make it play nice with ML stuff
- split this into two separate files and make them play nicely together

"""


# Global variables
# (It isn't good practice, but this is certainly better than passing them around like dead weight.)
input_file = "EventSets/TTbar_PU200_D49_20.root"
input_file_short = "ttbar_pu200"
output_dir = "TrkPlots/UprootPlots/"


def main(argv):
    """If first argument is the string 'length', print number of events in input file."""

    # Open ntuple, specify desired properties and cuts to be applied
    events = uproot_open(input_file)["L1TrackNtuple"]["eventTree"]
    properties_by_track_type = {"trk": ["pt", "eta", "nstub", "chi2rphi", "chi2rz", "genuine", "fake"],
                                "matchtrk": ["pt", "eta", "nstub", "chi2rphi", "chi2rz"],
                                "tp": ["pt", "eta", "nstub", "dxy", "d0", "eventid", "nmatch"]}
    cut_dicts = {"tp": {"eta": [-2.4, 2.4], "pt": [2, 100], "nstub": [4, 999],
                        "dxy": [-1.0, 1.0], "d0": [-1.0, 1.0], "eventid": 0}}

    if len(argv) == 1:
        plot(events, properties_by_track_type, cut_dicts)
    elif len(argv) == 2 and argv[1] == "length":
        track_type = properties_by_track_type.keys()[0]
        print(
            len(events[track_type + "_" + properties_by_track_type[track_type][0]]))


def plot(events, properties_by_track_type, cut_dicts):
    """Call functions from the file where all the plots are!

    Args:
            events:  an uproot event set
            properties_by_track_type:  properties by track type.
                    e.g., {"trk": ["pt", "eta", "nstub"]}
            cut_dicts:  a dict from track types
                    to dicts from properties to cut ranges.
                    e.g., {"tp": {"pt": [2, 100]}}
    """

    # Create ntuple properties dict from event set
    ntuple_properties = dict(map(lambda track_type, properties:
                                 (track_type, ntuple_to_dict(
                                     events, track_type, properties)),
                                 properties_by_track_type.keys(), properties_by_track_type.values()))
    ntuple_properties = cut_ntuple(ntuple_properties, cut_dicts)

    # Call appropriate plotting function(s)

    """
    ax = plot_roc_curve(ntuple_properties, "chi2rphi", [[0, 9999], [0, 500], [0, 100], [0, 70], [0, 50], [0, 40], [0, 30], [0, 20], [0, 10], [0, 5]], group_name="allpt", save_plot=True)
    axz = plot_roc_curve(ntuple_properties, "chi2rz", [[0, 9999], [0, 500], [0, 50], [0, 20], [0, 11], [0, 10], [0, 9], [0, 8], [0, 7], [0, 6], [0, 5]], group_name="allpt", save_plot=True)
    ntuple_properties = cut_ntuple(ntuple_properties, {"trk": {"pt": [8, 13000]}, "tp": {"pt": [8, 13000]}})
    plot_roc_curve(ntuple_properties, "chi2rphi", [[0, 9999], [0, 500], [0, 100], [0, 70], [0, 50], [0, 40], [0, 30], [0, 20], [0, 10], [0, 5]], group_name="pt8", ax=ax, color='orange')
    plot_roc_curve(ntuple_properties, "chi2rz", [[0, 9999], [0, 500], [0, 50], [0, 20], [0, 11], [0, 10], [0, 9], [0, 8], [0, 7], [0, 6], [0, 5]], group_name="pt8")#, ax=axz, color='orange')
    """

    genuine_properties = cut_trackset(ntuple_properties["trk"], {"genuine": 0})
    not_genuine_properties = cut_trackset(
        ntuple_properties["trk"], {"genuine": 1})
    fake_properties = cut_trackset(ntuple_properties["trk"], {"fake": 0})
    not_fake_properties_primary = cut_trackset(
        ntuple_properties["trk"], {"fake": 1})
    not_fake_properties_secondary = cut_trackset(
        ntuple_properties["trk"], {"fake": 2})

    overlay({"primary-vertex": not_fake_properties_primary["chi2rphi"],
             "secondary-vertex": not_fake_properties_secondary["chi2rphi"],
             "fake": fake_properties["chi2rphi"]}, "chi2rphi", "trk_fake_dist")
    overlay({"primary-vertex": not_fake_properties_primary["chi2rz"],
             "secondary-vertex": not_fake_properties_secondary["chi2rz"],
             "fake": fake_properties["chi2rz"]}, "chi2rz", "trk_fake_dist")
    overlay({"genuine": genuine_properties["chi2rphi"],
             "fake": not_genuine_properties["chi2rphi"]}, "chi2rphi", "trk_genuine_dist")
    overlay({"genuine": genuine_properties["chi2rz"],
             "fake": not_genuine_properties["chi2rz"]}, "chi2rz", "trk_genuine_dist")

    print("Process complete. Exiting program.")


main(argv)
