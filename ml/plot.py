import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import tensorflow as tf
from . import data
from .. import operations as ndops
from ..operations import select as sel


def true_positive_rate(labels, pred_labels,
        true_cond=sel(0), false_cond=sel(1)):
    """For a binary classifier label, returns the proportion of "true"
    cases that the model predicted correctly. Throws an error if the
    lists are of different sizes."""

    if len(labels) != len(pred_labels):
        raise ValueError("Predicted labels size differs from labels size")

    _, pred_labels_should_be_true = list(filter(lambda label, pred_label:
        true_cond(label),
        labels, pred_labels))

    return ndops.get_proportion_selected(
            pred_labels_should_be_true, true_cond)


def false_positive_rate(labels, pred_labels,
        true_cond=sel(0), false_cond=sel(1)):
    """For a binary classifier label, returns the proportion of "false"
    cases that the model predicted were "true". Throws an error if the
    lists are of different sizes."""
    
    if len(labels) != len(pred_labels):
        raise ValueError("Predicted labels size differs from labels size")

    _, pred_labels_should_be_false = list(filter(lambda label, pred_label:
        false_cond(label),
        labels, pred_labels))

    return ndops.get_proportion_selected(
            pred_labels_should_be_false, true_cond)


def apply_threshhold(pred_labels, threshhold):
    """Sends every prediction in the list below the threshhold
    (exclusive) to zero and everything above it (inclusive) to one."""

    return list(map(lambda pred: 1 if pred >= threshhold else 0, pred_labels))


def predict_labels(model, data, threshhold=None):
    """Run the model on each element of a dataset and produce a list of
    probabilistic predictions (note: not logits). Assumes a binary
    classifier. If a threshhold is provided, transforms the list into
    zeroes and ones accordingly.

    Args:
        model: a tensorflow or sklearn model capable of prediction
        data: a dataset for the model to run on
        threshhold: a number that will transform probabilities into
            concrete zero/one predictions. This should be between zero
            and one, but other values are allowed

    Returns:
        A list of probabilistic predictions
    """

    # Different models predict in different ways
    if "keras" in str(type(model)):
        pred_labels_full = tf.keras.Sequential(
                [model, tf.keras.layers.Softmax()]).predict(data)
        pred_labels = list(map(lambda l: l[0], pred_labels_full))
    else:
        pred_labels = np.sum(model.predict_proba(data), axis=1)

    return pred_labels if threshhold is None \
            else apply_threshhold(pred_labels, threshhold)


def create_roc(data, labels, models, model_names,
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
            predicted. Used to cut if cuts is true
    """

    # Create roc curves from models' predictions on data and labels
    for model, model_name in zip(models, model_names):
        pred_labels = predict_labels(model, data)
        fpr, tpr, _ = roc_curve(labels, pred_labels)
        auc = roc_auc_score(labels, pred_labels)
        auc_string = ' ('+str(round(auc,3))+')'
        plt.plot(fpr, tpr, label=model_name+auc_string,
                linewidth=2)

    # Plot against sets of cuts, if any are given
    for cut in cuts:
        track_prop_dict = data.make_track_prop_dict_from_dataset(
               data, labels, data_properties, label_property)
        cut_indices = ndops.select_indices(track_prop_dict, cut)
        cut_pred_labels = list(map(
            lambda index: 1 if index in cut_indices else 0,
           range(ndops.track_prop_dict_length(track_prop_dict))))
        fpr_cut = false_positive_rate(labels, cut_pred_labels) 
        tpr_cut = true_positive_rate(labels, cut_pred_labels)
        plt.scatter(fpr_cut, tpr_cut,
                s=80, marker='*', label='cuts', color='red')
        
    plt.tick_params(labelsize=14)
    plt.xlabel('FPR', fontsize=20)
    plt.ylabel('TPR', fontsize=20)
    #plt.xlim(0, .3)
    #plt.ylim(.9, 1)
    plt.legend(loc='best',fontsize=14)
    plt.savefig("roc_curve.pdf")

