import uproot
from sys import argv
import numpy as np
from copy import deepcopy
from math import sqrt
import matplotlib as mpl
import matplotlib.pyplot as plt


"""
UPROOT PLOT
TODO: what is uproot plot anyway?
"""

def main(argv):
	"""If first argument is the string 'length', print length of input file."""

	# Make ntuple properties dict
	events = uproot.open(input_file)["L1TrackNtuple"]["eventTree"]
	properties_by_track_type = {"trk": ["pt", "eta", "nstub", "chi2rphi", "chi2rz", "genuine", "fake"],
                "matchtrk": ["pt", "eta", "nstub", "chi2rphi", "chi2rz"],
                "tp": ["pt", "eta", "nstub"]}
	cut_dicts={"tp": {"eta": [-2.4, 2.4], "pt": [4, 13000], "nstub": [4, 8]},
		"trk": {"eta": [-2.4, 2.4], "pt": [4, 13000], "nstub": [4, 8]}}

	if len(argv) == 1:
		plot(events, properties_by_track_type, cut_dicts)
	elif len(argv) == 2 and argv[1] == "length":
		track_type = properties_by_track_type.keys()[0]
		print(len(events[track_type + "_" + properties_by_track_type[track_type][0]]))


def plot(events, properties_by_track_type, cut_dicts):
	"""Get data, then call functions to make graphs from that data.

	Args:
		events:  an uproot event set
		properties_by_track_type:  properties by track type.
			e.g., {"trk": ["pt", "eta", "nstub"]}
		cut_dicts:  a dict from track types
			to dicts from properties to cut ranges.
			e.g., {"tp": {"pt": [2, 100]}}
	"""

        ntuple_properties = dict(map(lambda track_type, properties: (track_type, ntuple_to_dict(events, track_type, properties)), properties_by_track_type.keys(), properties_by_track_type.values()))
	ntuple_properties = cut_ntuple(ntuple_properties, cut_dicts)

	# Call appropriate plotting function(s)

        #print(eff_from_matchtrks_and_tps(ntuple_properties["matchtrk"]["pt"], ntuple_properties["tp"]["pt"], lambda mt_pt: mt_pt > 8, lambda tp_pt: tp_pt > 8))

	#proportion_foreach_bin(ntuple_properties["trk"]["genuine"], ntuple_properties["trk"]["eta"], lambda trk_genuine: trk_genuine == 0)

	#ax = plot_roc_curve(ntuple_properties, "chi2rphi", [[0, 9999], [0, 500], [0, 100], [0, 70], [0, 50], [0, 40], [0, 30], [0, 20], [0, 10], [0, 5]], group_name="allpt", save_plot=True)
	#axz = plot_roc_curve(ntuple_properties, "chi2rz", [[0, 9999], [0, 500], [0, 50], [0, 20], [0, 11], [0, 10], [0, 9], [0, 8], [0, 7], [0, 6], [0, 5]], group_name="allpt", save_plot=True)
	#ntuple_properties = cut_ntuple(ntuple_properties, {"trk": {"pt": [8, 13000]}, "tp": {"pt": [8, 13000]}})
	#plot_roc_curve(ntuple_properties, "chi2rphi", [[0, 9999], [0, 500], [0, 100], [0, 70], [0, 50], [0, 40], [0, 30], [0, 20], [0, 10], [0, 5]], group_name="pt8", ax=ax, color='orange')
	#plot_roc_curve(ntuple_properties, "chi2rz", [[0, 9999], [0, 500], [0, 50], [0, 20], [0, 11], [0, 10], [0, 9], [0, 8], [0, 7], [0, 6], [0, 5]], group_name="pt8")#, ax=axz, color='orange')

	genuine_properties = cut_trackset(ntuple_properties["trk"], {"genuine": 0})
	not_genuine_properties = cut_trackset(ntuple_properties["trk"], {"genuine": 1})
	fake_properties = cut_trackset(ntuple_properties["trk"], {"fake": 0})
	not_fake_properties_primary = cut_trackset(ntuple_properties["trk"], {"fake": 1})
	not_fake_properties_secondary = cut_trackset(ntuple_properties["trk"], {"fake": 2})

	print(genuine_properties)
	print(not_genuine_properties)
	print(fake_properties)
	print(not_fake_properties_primary)
	print(not_fake_properties_secondary)

	overlay({"primary-vertex": not_fake_properties_primary["chi2rphi"],
		"secondary-vertex": not_fake_properties_secondary["chi2rphi"],
		"fake": fake_properties["chi2rphi"]}, "chi2rphi", "trk_fake_dist")
	overlay({"primary-vertex": not_fake_properties_primary["chi2rz"],
		"secondary-vertex": not_fake_properties_secondary["chi2rz"],
		"fake": fake_properties["chi2rz"]}, "chi2rz", "trk_fake_dist")
	overlay({"genuine": genuine_properties["chi2rphi"], "fake": not_genuine_properties["chi2rphi"]}, "chi2rphi", "trk_genuine_dist")
	overlay({"genuine": genuine_properties["chi2rz"], "fake": not_genuine_properties["chi2rz"]}, "chi2rz", "trk_genuine_dist")

	print("Process complete. Exiting program.")


def ntuple_to_dict(events, track_type, properties):
	"""Takes in an uproot ntuple, the data type, and properties to be extracted;
	returns a dictionary from a property name to flattened array of values.
	Note that due to this flattening, all information about which tracks are
	from which event is lost.

	Args:
		events:  an uproot event set
		track_type:  trk, matchtrk, etc.
		properties:  pt, eta, pdgid, etc.

	Returns:
		For a particular track type, a dict from properties to values
	"""

	tracks_properties = {}
	for property in properties:
		tracks_properties[property] = list(events[track_type + "_" + property].array().flatten())

	return tracks_properties


def cut_ntuple(ntuple_properties,
	cut_dicts={"tp": {"eta": [-2.4, 2.4], "pt": [4, 13000], "nstub": [4, 8]},
		"trk": {"eta": [-2.4, 2.4], "pt": [4, 13000], "nstub": [4, 8]}}):
	"""Takes in this file's representation of an ntuple (a dict from track types
	to dicts from properties to value lists) and cuts each trackset. Takes into
	account that matchtrks and tps must be cut together, and that trks and matchtps
	must be cut together as well."""

	# Build list of tracks to cut from tp/matchtrk group and trk/matchtp groups
	cut_indices_dict = {"trk": [], "matchtrk": [], "tp": [], "matchtp": []}
	cut_indices_dict.update(dict(map(lambda track_type, cut_dict:
		(track_type, get_indices_meeting_conditions(ntuple_properties[track_type], cut_dict)),
		cut_dicts.keys(), cut_dicts.values())))

	# Combine trk and matchtp, tp and matchtrk indices to remove (respectively), sorting and removing duplicates
	trk_matchtp_indices_to_cut = sorted(list(dict.fromkeys(cut_indices_dict["trk"] + cut_indices_dict["matchtp"])))
	tp_matchtrk_indices_to_cut = sorted(list(dict.fromkeys(cut_indices_dict["tp"] + cut_indices_dict["matchtrk"])))

	cut_ntuple_properties = {}
	for track_type, trackset in ntuple_properties.items():
		if track_type in ["trk", "matchtp"]:
			indices_to_cut = trk_matchtp_indices_to_cut
		if track_type in ["tp", "matchtrk"]:
			indices_to_cut = tp_matchtrk_indices_to_cut
		cut_ntuple_properties[track_type] = cut_trackset_by_indices(ntuple_properties[track_type], indices_to_cut)

	return cut_ntuple_properties


def cut_trackset(tracks_properties, cut_dict={},
		basic_cut_dict={"eta": [-2.4, 2.4], "pt": [4, 13000], "nstub": [4, 8]}):
	"""Cuts a trackset (trk_XYZ, tp_XYZ, etc.) given the cut dicts."""

	basic_cut_dict.update(cut_dict)  # cut dict must override basic_cut_dict elements

	return cut_trackset_by_indices(tracks_properties,
		get_indices_meeting_conditions(tracks_properties, basic_cut_dict))


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
	supposed_num_tracks = len(tracks_properties.values()[0])
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
	have to be sorted by size."""

	# Copy, then delete all tracks (going backwards so as not to affect indices)
	cut_tracks_properties = deepcopy(tracks_properties)
	for track_to_cut in reversed(sorted(indices_to_cut)):
		for property in cut_tracks_properties.keys():
			del cut_tracks_properties[property][track_to_cut]
	
	return cut_tracks_properties


def get_proportion_meeting_condition(tracks_property, property_condition):
	"""Find the proportion of tracks whose given property meets a condition.
	If the number of tracks is zero, returns zero."""

	if len(tracks_property) == 0:
		print("Cannot calculate proportion meeting condition in zero-length quantity. Returning zero.")
		return 0
	else:
		return float(sum(map(property_condition, tracks_property))) / len(tracks_property)


def eff_from_matchtrks_and_tps(matchtrks_prop, tps_prop,
		matchtrk_cond=lambda trk_prop_val: trk_prop_val != -999,
		tp_cond=lambda tp_prop_val: True):
        """Finds the efficiency from any given matchtrk property and the number
	of tracking particles. Subtracts out filler values from total matchtrks."""

	print(len(filter(matchtrk_cond, matchtrks_prop)))
	print(len(filter(tp_cond, tps_prop)))

	return float(len(filter(matchtrk_cond, matchtrks_prop))) / len(filter(tp_cond, tps_prop))
 

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

	# We don't want to alter ntuple_properties, do we?
	ntuple_properties = deepcopy(ntuple_properties_in)

	# Build up cuts plot info, which is what will be plotted
	cuts_plot_info = {"cut": [], "eff": [], "fake_rate": []}
	trackset_to_cut = {}	
	cut_tracks_properties = ntuple_properties["trk"]
	cut_matchtracks_properties = ntuple_properties["matchtrk"]
	for cut in cuts:
		trackset_to_cut["trk"] = cut_tracks_properties if cuts_increasing else ntuple_properties["trk"]
		trackset_to_cut["matchtrk"] = cut_matchtracks_properties if cuts_increasing else ntuple_properties["matchtrk"]
		cut_tracks_properties = cut_trackset(trackset_to_cut["trk"], {cut_property: cut})
		cut_matchtracks_properties = cut_trackset(trackset_to_cut["matchtrk"], {cut_property: cut})
		cuts_plot_info["cut"].append(cut[1])  # only get upper bound of cut
		cuts_plot_info["eff"].append(eff_from_matchtrks_and_tps(cut_matchtracks_properties["pt"], ntuple_properties["tp"]["pt"]))
		cuts_plot_info["fake_rate"].append(get_proportion_meeting_condition(cut_tracks_properties["genuine"], lambda gen_track: gen_track == 0))

	# Now plot!
	ax.plot(cuts_plot_info["fake_rate"], cuts_plot_info["eff"], "b.", label=group_name, color=color) 

	if save_plot:
		ax.set_xlabel("Track fake rate")
		ax.set_ylabel("Tracking efficiency")
		ax.legend()
		#for fake_rate, eff, cut in zip(cuts_plot_info["fake_rate"], cuts_plot_info["eff"], cuts_plot_info["cut"]):
			#ax.annotate(str(cut), ax=(fake_rate, eff))
		ax.set_title("Fake rate and efficiency for changing " + cut_property + " cuts " + group_name)
		plt.savefig(output_dir + "eff_vs_fakerate_cuts_" + cut_property + "_" + group_name + ".pdf")

	return ax


def proportion_foreach_bin(trks_property, trks_property_wrt, property_condition, num_bins=30):
        """wrt meaning "with respect to" - bin tracks by property, and then see how another
	property of those tracks meets a certain condition. For example, fake rate by eta,
	or efficiency by pT."""  # TODO support efficiency

	bins = np.linspace(min(trks_property_wrt), max(trks_property_wrt), num_bins)

	# Sort tracks into bins
	bins_trks_property = [ [] for bin in bins[:-1] ]
	for trk_property, trk_property_wrt in zip(trks_property, trks_property_wrt):
                for bindex, bin in zip(reversed(range(len(bins))), reversed(bins)):  # reversed iteration makes checking values against lower bin edges easier
			if trk_property_wrt > bin:
				bins_trks_property[bindex].append(trk_property)
				break

	# For each bin, find tracks meeting condition, get ratio and stdev
	bins_trks_proportions = []
	bins_trks_proportions_stdev = []
	for bin_trks_property in bins_trks_property:
		bins_trks_proportions.append(get_proportion_meeting_condition(bin_trks_property, property_condition))
		bins_trks_proportions_stdev.append(1/sqrt(len(bin_trks_property)) if len(bin_trks_property) > 0 else 0)  #TODO generalize: but also, is this alright?

	plt.errorbar(x=[ (bin_low_edge + bin_high_edge) / 2 for bin_low_edge, bin_high_edge in zip(bins[:-1], bins[1:]) ], 
			y=bins_trks_proportions, yerr=bins_trks_proportions_stdev,
			elinewidth=1, drawstyle="steps-mid")
	plt.xlabel("")
	plt.ylabel("")
	#plt.legend()
	plt.title("Geez scoob I hope this works")
	plt.savefig(output_dir + "geez.pdf")


def overlay(trackset_dict, overlaid_property="", group_name="", num_bins=30):
	"""Takes in a dictionary where each value is a list of values of overlaid_property
	and each key identifies what makes its value set unique. For example, the dict
	could map from the string "fake" to the list of overlaid_properties values from
	fake tracks, and does the same for real tracks."""

	_, bins, _ = plt.hist(trackset_dict.values()[0], bins=num_bins, log=True, density=True, histtype="step", label=trackset_dict.keys()[0])
	for trackset_name, trackset in trackset_dict.items()[1:]:
		plt.hist(trackset, bins=bins, log=True, density=True, histtype="step", label=trackset_name)
	plt.xlabel(overlaid_property)
	plt.ylabel("# of tracks")
	plt.legend()
	plt.title(r"Normalized " + overlaid_property + " values for tracks," + input_file_short + " " + group_name)
	plt.savefig(output_dir + input_file_short + "_" + overlaid_property + "_" + group_name + ".pdf")
	plt.clf()


def hist2D_properties(property1, property2, property1name="", property2name="", group_name="", bins=30):
	"""Make a 2D histogram with property 1 on the x axis and property 2 on the y axis."""

	cmap = mpl.cm.viridis
	cmap.set_under("w")

	plt.hist2d(property1, property2, bins=bins, norm=mpl.colors.LogNorm(vmin=.5))
	plt.colorbar()
	plt.xlabel(property1name)
	plt.ylabel(property2name)
	plt.title(property1 + " vs. " + property2 + " for " + group_name + " " + input_file_short + " tracks")
	plt.savefig(output_dir + input_file_short + "_" + property1 + "_vs_" + property2 + "_" + group_name + ".pdf")
	plt.clf()


# Global variables
input_file = "EventSets/TTbar_PU200_D49.root"
input_file_short = "ttbar_pu200"
output_dir = "TrkPlots/UprootPlots/"

main(argv)

