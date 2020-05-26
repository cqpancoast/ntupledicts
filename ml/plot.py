from numpy import linspace
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from .. import plot as ndplot
from . import predict as mlpred


def plot_pred_comparison_by_track_property(dataset, pred_name,
        pred_comparison, bin_property, bins=10, threshold=.6,
        legend_id=None, ax=None):
    """Compares true labels to the model predictions by some function,
    binned by a track property present in data.

    Args:
        dataset: a TrackPropertiesDataset.
        pred_name: the name of a prediction to be found in dataset
        pred_comparison: a function that takes in the labels, the
            predicted labels, and a threshold value, and returns a
            number measuring some property of the predicted labels'
            relation to the actual ones.
        bin_property: a property in data_properties or the
            label_property that will split the dataset into bins.
        bins: either an int for the number of bins, a 3-tuple of the
            form (low_bound, high_bound, num_bins), or a list of
            numbers. See ntupledict.operations.make_bins() for info.
        threshold: the limit at which a prediction signifies one or
            the other value of a binary classification,
        legend_id: the entry in the legend for the line to be plotted.
            Calling ax.legend() should be done outside this function.
        ax: an axes object to be used to plot in this function.

    Returns:
        The Axes object to be used to plot in this function.
    """

    if ax is None:
        ax = plt.figure().add_subplot(111)

    track_prop_dict = dataset.to_track_prop_dict(include_preds=True)

    def measure_pred_comparison(track_prop_dict):
        """Measures the prediction comparison of true labels and
        predicted labels that are within a track propeties dict."""

        return pred_comparison(track_prop_dict[dataset.get_label_property()],
                track_prop_dict[pred_name],
                threshold)

    return ndplot.plot_measure_by_bin(track_prop_dict, bin_property,
            measure_pred_comparison, bins, legend_id, ax)


def plot_pred_comparison_by_threshold(dataset, pred_name,
        pred_comparison, thresholds=10, legend_id=None,
        ax=None):
    """Compares true labels to the model predictions by some function
    at various thresholds.

    Args:
        dataset: a TrackPropertiesDataset.
        pred_name: the name of a prediction to be found in dataset.
        pred_comparison: a function that takes in the labels, the
            predicted labels, and a threshold value, and returns a
            number measuring some property of the predicted labels'
            relation to the actual ones.
        thresholds: the limits at which a prediction signifies one or
            the other value of a binary classification.
        legend_id: the entry in the legend for the line to be plotted.
            Calling ax.legend() should be done outside this function.
        ax: an axes object to be used to plot in this function.

    Returns:
        The axes object to used to plot in this function.
    """

    if ax is None:
        ax = plt.figure().add_subplot(111)

    # Generate threshold list if thresholds is not a list
    if not isinstance(thresholds, list):
        thresholds = linspace(0, 1, thresholds)

    ax.scatter(thresholds, list(map(lambda threshold:
        pred_comparison(dataset.get_labels(),
            dataset.get_prediction(pred_name), threshold),
        thresholds)), label=legend_id)
    ax.set_xlabel("Decision threshold")

    return ax


def plot_rocs(dataset, prob_pred_names=[], def_pred_names=[],
              xlims=(0, .3), ylims=(.9, 1)):
    """Create ROC curves through true positive rate / false positive
    rate space for different models by changing the cut on predictions.
    Note that this only works if the label is from a binary classifier
    such as trk_genuine.

    Args:
        dataset: a TrackPropertiesDataset containing the data, labels,
            and corresponding property names for both.
        prob_pred_names: names of probablistic predictions accessible
            from dataset. Typically the names of the models that made
            them. Will be plotted as a curve.
        def_pred_names: names of pre-thresholded predictions
            accessible from the dataset. This is what is used for cut-
            generated predictions. Plotted as a point.
    """

    ax = plt.figure().add_subplot(111)

    # Plot ROC curve for models
    for prob_pred_name in prob_pred_names:
        pred_prob_labels = dataset.get_prediction(prob_pred_name)
        fpr, tpr, _ = roc_curve(dataset.get_labels(), pred_prob_labels)
        auc = roc_auc_score(dataset.get_labels(), pred_prob_labels)
        auc_string = " ({})".format(str(round(auc, 3)))
        ax.plot(fpr, tpr, label=prob_pred_name+auc_string,
                linewidth=2)

    # Plot cuts, if any are given
    for def_pred_name in def_pred_names:
        pred_labels = dataset.get_prediction(def_pred_name)
        fpr_cut = mlpred.false_positive_rate(dataset.get_labels(), pred_labels)
        tpr_cut = mlpred.true_positive_rate(dataset.get_labels(), pred_labels)
        ax.scatter(fpr_cut, tpr_cut,
                   s=80, marker="*", label="cuts", color="red")

    ax.tick_params(labelsize=14)
    ax.set_xlabel("FPR", fontsize=20)
    ax.set_ylabel("TPR", fontsize=20)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.legend(loc="best", fontsize=14)

    return ax
