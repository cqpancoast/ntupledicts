import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from numpy import linspace
from .data import make_track_prop_dict_from_dataset as to_track_prop_dict
from .. import operations as ndops
from .. import plot as ndplot
from . import predict as mlpred


def plot_pred_comparison_by_track_property(model, data, labels,
        data_properties, label_property, pred_comparison, bin_property, 
        bins=30, threshhold=.5, ax=plt.figure().add_subplot(111)):
    """Compares true labels to the model predictions by some function,
    binned by a track property present in data.

    Args:
        model: a model that can take in data and output predictions
        data: the data associated with the labels given to the model
        labels: a list of binary classifier values, either zero or one
        data_properties: the properties of each track in the data set.
            Used to cut if cuts is true
        label_property: the property of the data label that is being
            predicted. Used to cut if cuts has any elements
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

    pred_labels = mlpred.predict_labels(model, data)
    
    # Add predictive labels as part of track properties dict
    predkey = "pred_" + label_property
    track_prop_dict = to_track_prop_dict(data, labels,
            data_properties, label_property)
    track_prop_dict.update({predkey: pred_labels})

    def measure_pred_comparison(track_prop_dict):
        """Measures the prediction comparison of true labels and
        predicted labels that are within a track propeties dict."""

        return pred_comparison(track_prop_dict[label_property],
                track_prop_dict[predkey],
                threshhold)

    return ndplot.plot_measure_by_bin(track_prop_dict, bin_property,
            measure_pred_comparison, bins, ax)


def plot_pred_comparison_by_threshhold(model, data, labels,
        pred_comparison, threshholds=30, ax=plt.figure().add_subplot(111)):
    """Compares true labels to the model predictions by some function 
    at various threshholds.

    Args:
        model: a model that can take in data and output predictions
        data: the data associated with the labels given to the model
        labels: a list of binary classifier values, either zero or one
        pred_comparison: a function that takes in the labels, the
            predicted labels, and a threshhold value, and returns a
            number measuring some property of the predicted labels'
            relation to the actual ones
        threshholds: the limits at which a prediction signifies one or
            the other value of a binary classification
        ax: an axes object to be used to plot this function

    Returns:
        The axes object to be used to plot this function
    """

    pred_labels = mlpred.predict_labels(model, data)

    # Generate threshhold list if threshholds is not a list
    if not isinstance(threshholds, list):
        threshholds = linspace(0, 1, threshholds)

    ax.scatter(threshholds, list(map(lambda threshhold: 
        pred_comparison(labels, pred_labels, threshhold),
        threshholds)))
    ax.set_xlabel("Decision Threshhold")

    return ax
    

def plot_rocs(data, labels, models, model_names,
        cuts=[], data_properties=None, label_property=None):
    """Create ROC curves through true positive rate / false positive
    rate space for different models by changing the cut on model-
    generated predications. Optionally, plot these against a set of 
    cuts. Note that this only works if the label is from a binary
    classifier such as trk_genuine.

    Args:
        data: a tensorflow tensor of data
        labels: a tensorflow tensor of label data
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
        pred_labels = mlpred.predict_labels(model, data)
        fpr, tpr, _ = roc_curve(labels, pred_labels)
        auc = roc_auc_score(labels, pred_labels)
        auc_string = ' ('+str(round(auc,3))+')'
        ax.plot(fpr, tpr, label=model_name+auc_string,
                linewidth=2)

    # Plot cuts, if any are given
    for cut in cuts:
        track_prop_dict = to_track_prop_dict(
               data, labels, data_properties, label_property)
        cut_indices = ndops.select_indices(track_prop_dict, cut)
        cut_pred_labels = list(map(
            lambda index: 0 if index in cut_indices else 1,
           range(ndops.track_prop_dict_length(track_prop_dict))))
        fpr_cut = mlpred.false_positive_rate(labels, cut_pred_labels)
        tpr_cut = mlpred.true_positive_rate(labels, cut_pred_labels)
        ax.scatter(fpr_cut, tpr_cut,
                s=80, marker='*', label='cuts', color='red')
        
    ax.tick_params(labelsize=14)
    ax.set_xlabel('FPR', fontsize=20)
    ax.set_ylabel('TPR', fontsize=20)
    #ax.xlim(0, .3)
    #ax.ylim(.9, 1)
    ax.legend(loc='best',fontsize=14)

    return ax

