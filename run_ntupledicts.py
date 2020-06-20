from uproot import open as uproot_open
from src import ntupledicts
from ntupledicts.load import root_files_to_ntuple_dict
from ntupledicts import operations as ndops
from ntupledicts import analyze as ndanl
from ntupledicts.operations import select as sel
from ntupledicts import plot as ndplot
from ntupledicts.ml import data as ndmldata
from ntupledicts.ml import predict as ndmlpred
from ntupledicts.ml import models as ndmlmodels
from ntupledicts.ml import plot as ndmlplot
from matplotlib.pyplot import cla, sca, gca, savefig, xticks
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Dense
from time import time



# I/O global variables
input_files = ["eventsets/D49_ZEE.root", "eventsets/D49_ZMM.root", "eventsets/D49_QCD.root"]
output_dir = "/Users/caseypancoast/Desktop/plotbbs/"


def main():

    # Open ntuples, specify desired properties and cuts to be applied
    properties_by_track_type = {"trk": ["pt", "eta", "z0", "nstub", "chi2", "bendchi2", "genuine",
                                            "matchtp_pdgid", "hitpattern"]}

    ntuple_dict = root_files_to_ntuple_dict(input_files, properties_by_track_type,
            keep_invalid_vals=False)
    # go(ntuple_dict)
    test(ntuple_dict["trk"])

    print("Process complete. Exiting program.")


def go(ntuple_dict):

    # Count layers meeting these conditions for each track. (see ntupledicts.analyze for documentation)
    missing_2S_layer = lambda expected, hit, ps_2s: not ps_2s and expected and not hit
    missing_PS_layer = lambda expected, hit, ps_2s: ps_2s and expected and not hit

    ntuple_dict["trk"]["missing2S"] = ndanl.create_stub_info_list(ntuple_dict["trk"],
            ndanl.basic_process_stub_info(missing_2S_layer))
    ntuple_dict["trk"]["missingPS"] = ndanl.create_stub_info_list(ntuple_dict["trk"],
            ndanl.basic_process_stub_info(missing_PS_layer))

    # Make datasets
    train_ds, eval_ds, test_ds = ndmldata.TrackPropertiesDataset(ntuple_dict["trk"],
            "genuine", ["pt", "chi2", "bendchi2", "nstub"]).split([.7, .2, .1])

    # Train models on dataset
    # NN = ndmlmodels.make_neuralnet(train_ds, eval_dataset=eval_ds, hidden_layers=[15, 8], epochs=100)
    GBDT = ndmlmodels.make_gbdt(train_ds)
    cuts = [{"chi2rphi": sel(0, 23), "chi2rz": sel(0, 7), "chi2": sel(0, 21)}]

    # test_ds.add_prediction("NN", ndmlpred.predict_labels(NN, test_ds.get_data()))
    test_ds.add_prediction("GBDT", ndmlpred.predict_labels(GBDT, test_ds.get_data()))
    test_ds.add_prediction("cuts", ndmlpred.predict_labels_cuts(next(iter(cuts)), test_ds))

    # plot(test_ds, {"GBDT": GBDT, "cuts": cuts})
    test(test_ds)


def test(track_prop_dict):
    """Use this function when testing samples or functionality rather
    than actually running stuff."""

    # TODO determine which method is faster for making cut predictions (see ndmlpred)
    tpds = ndmldata.TrackPropertiesDataset(track_prop_dict, "genuine")
    ndmlpred.predict_labels_cuts({"chi2": sel(0, 20)}, tpds)


main()

