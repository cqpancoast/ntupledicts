import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from numpy import linspace
from .data import make_track_prop_dict_from_dataset as to_track_prop_dict
#from .. import operations as ndops
from .. import plot as ndplot
from . import predict as mlpred


def plot_pred_comparison_by_track_property(pred_labels, dataset,
        pred_comparison, bin_property, bins=10, threshhold=.5,
        ax=plt.figure().add_subplot(111)):
    """Compares true labels to the model predictions by some function,
    binned by a track property present in data.

    Args:
        pred_labels: a list of labels predicted by some model or cuts
            using data from dataset
        dataset: a TrackPropertiesDataset containing the data, labels,
            and corresponding property names for both
        pred_comparison: a function that takes in the labels, the
            predicted labels, and a threshhold value, and returns a
            number measuring some property of the predicted labels'
            relation to the actual ones
        bin_property: a property in data_properties or the
            label_property that will split the dataset into bins
        bins: either an int for the number of bins, a 3-tuple of the
            form (low_bound, high_bound, num_bins), or a list of
            numbers. See ntupledict.operations.make_bins() for info
        threshhold: the limit at which a prediction signifies one or
            the other value of a binary classification
        ax: an axes object to be used to plot this function

    Returns:
        The axes object to be used to plot this function
    """

    # Add predictive labels as part of track properties dict
    predkey = "pred_" + dataset.label_property
    track_prop_dict = to_track_prop_dict(dataset)

    track_prop_dict.update({predkey: pred_labels})

    def measure_pred_comparison(track_prop_dict):
        """Measures the prediction comparison of true labels and
        predicted labels that are within a track propeties dict."""

        return pred_comparison(track_prop_dict[dataset.label_property],
                track_prop_dict[predkey],
                threshhold)

    return ndplot.plot_measure_by_bin(track_prop_dict, bin_property,
            measure_pred_comparison, bins, ax)


def plot_pred_comparison_by_threshhold(pred_labels, labels,
        pred_comparison, threshholds=10, ax=plt.figure().add_subplot(111)):
    """Compares true labels to the model predictions by some function 
    at various threshholds.

    Args:
        model: a model that can take in data and output predictions
        dataset: a TrackPropertiesDataset containing the data, labels,
            and corresponding property names for both
        pred_comparison: a function that takes in the labels, the
            predicted labels, and a threshhold value, and returns a
            number measuring some property of the predicted labels'
            relation to the actual ones
        threshholds: the limits at which a prediction signifies one or
            the other value of a binary classification
        ax: an axes object to be used to plot this function

    Returns:
        The axes object to used to plot this function
    """

    # Generate threshhold list if threshholds is not a list
    if not isinstance(threshholds, list):
        threshholds = linspace(0, 1, threshholds)

    ax.scatter(threshholds, list(map(lambda threshhold: 
        pred_comparison(labels, pred_labels, threshhold),
        threshholds)))
    ax.set_xlabel("Decision Threshhold")

    return ax
    

def plot_rocs(dataset, models, model_names,
        cuts=[], data_properties=None, label_property=None):
    """Create ROC curves through true positive rate / false positive
    rate space for different models by changing the cut on model-
    generated predications. Optionally, plot these against a set of 
    cuts. Note that this only works if the label is from a binary
    classifier such as trk_genuine.

    Args:
        dataset: a TrackPropertiesDataset containing the data, labels,
            and corresponding property names for both
        models: a list of models with predictive capabilites
        model_names: the names of the models for the plot legend
        cuts: an optional list of selector dictionaries to apply to
            the data to predict the binary variable in question
        data_properties: the properties of each track in the data set.
            Used to cut if cuts is true
        label_property: the property of the data label that is being
            predicted. Used to cut if cuts has any elements
    """

    ax = plt.figure().add_subplot(111)

    # Plot ROC curve for models 
    for model, model_name in zip(models, model_names):
        pred_labels = mlpred.predict_labels(model, dataset.data)
        fpr, tpr, _ = roc_curve(dataset.labels, pred_labels)
        auc = roc_auc_score(dataset.labels, pred_labels)
        auc_string = ' ('+str(round(auc,3))+')'
        ax.plot(fpr, tpr, label=model_name+auc_string,
                linewidth=2)

    # Plot cuts, if any are given
    for cut in cuts:
        pred_labels = mlpred.predict_labels_cuts(cut, dataset)
        fpr_cut = mlpred.false_positive_rate(dataset.labels, pred_labels)
        tpr_cut = mlpred.true_positive_rate(dataset.labels, pred_labels)
        ax.scatter(fpr_cut, tpr_cut,
                s=80, marker='*', label='cuts', color='red')
        
    ax.tick_params(labelsize=14)
    ax.set_xlabel('FPR', fontsize=20)
    ax.set_ylabel('TPR', fontsize=20)
    #ax.xlim(0, .3)
    #ax.ylim(.9, 1)
    ax.legend(loc='best',fontsize=14)

    return ax

