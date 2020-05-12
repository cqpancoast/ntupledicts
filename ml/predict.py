from numpy import sum as npsum
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Softmax
from .. import operations as ndops
from ..operations import select as sel


def check_pred_labels_size(labels, pred_labels):
    """Raises a ValueError is the predicted labels are a different size
    than the actual labels. Otherwise, does nothing."""

    if len(labels) != len(pred_labels):
        raise ValueError("Predicted labels size differs from labels size")


def apply_threshhold(pred_prob_labels, threshhold):
    """Sends every prediction in the list below the threshhold
    (exclusive) to zero and everything above it (inclusive) to one.
    In the parlance of this file, turns predicted probablistic labels
    into predicted labels."""

    return list(map(lambda pred: 1 if pred >= threshhold else 0,
        pred_prob_labels))


def pred_proportion_given_truth_case(labels, pred_labels,
        labels_restriction, pred_labels_case, threshhold=.5):
    """Look at the relative proportion of a value of the predicted
    probability labels, looking only at values who match to an acutal
    label of a particular case.

    This is the generalization of true and false positive rates.

    Args:
        labels: a list of true binary classifer labels
        pred_labels: a list of predicted binary classifier labels,
            OR a list of probablistic predictions to be converted to
            exact predictions using the threshhold
        labels_restriction: a function returning true for values of the
            label case to which the domain should be restricted
        pred_labels_case: a function returning true for values to count
            part of the proportion in the prediction with restricted
            domain
        threshhold: a threshhold to apply to the probablistic data
            before computing the agreement. Assumes binary classifier

    Returns:
        The proportion of predicted values meeting a certain case given
        a restriction of true values meeting a certain case.

    Raises:
        ValueError if the true and predicted labels differ in size
    """

    check_pred_labels_size(labels, pred_labels)

    labels_and_preds = list(map(lambda label, pred_label:
        {"label": label, "pred": pred_label},
        labels, pred_labels))

    labels_and_preds_restricted_domain = list(filter(lambda label_and_pred:
        labels_restriction(label_and_pred["label"]),
        labels_and_preds))

    pred_labels_restricted_domain = list(map(lambda label_and_pred:
        label_and_pred["pred"],
        labels_and_preds_restricted_domain))

    pred_labels_threshholded_rest_dom = apply_threshhold(
            pred_labels_restricted_domain, threshhold)

    return ndops.get_proportion_selected(pred_labels_threshholded_rest_dom,
            pred_labels_case)


def true_positive_rate(labels, pred_labels, threshhold=.5):
    """For a binary classifier label, returns the proportion of "true"
    cases that the model predicted correctly. Throws an error if the
    lists are of different sizes.

    Args:
        labels: a list of binary classifier labels
        pred_labels: a list of predicted binary classifier labels,
            OR a list of probablistic predictions to be converted to
            exact predictions using the threshhold
        threshhold: a threshhold to apply to the probablistic data

    Returns:
        The proportion of "true" cases that a model predicted correctly

    Raises:
        ValueError if the true and predicted labels differ in size
    """

    return pred_proportion_given_truth_case(labels, pred_labels,
            sel(1), sel(1), threshhold)


def false_positive_rate(labels, pred_labels, threshhold=.5):
    """For a binary classifier label, returns the proportion of "false"
    cases that the model predicted were "true". Raises an error if the
    lists are of different sizes.

    Args:
        labels: a list of binary classifier labels
        pred_labels: a list of predicted binary classifier labels,
            OR a list of probablistic predictions to be converted to
            exact predictions using the threshhold
        threshhold: a threshhold to apply to the probablistic data

    Returns:
        The proportion of "false" cases that a model predicted "true" 

    Raises:
        ValueError if the true and predicted labels differ in size
    """
    
    return pred_proportion_given_truth_case(labels, pred_labels,
            sel(0), sel(1), threshhold)


#TODO something is UP here...
def predict_labels(model, data):
    """Run the model on each element of a dataset and produce a list of
    probabilistic predictions (note: not logits). Assumes a binary
    classifier. Does not apply a threshhold.

    Args:
        model: a tensorflow or sklearn model capable of prediction
        data: a dataset for the model to run on

    Returns:
        A Python list of probabilistic predictions
    """

    # Different models predict in different ways
    if "keras" in str(type(model)):
        #pred_prob_labels_full = Sequential(
        #        [model, Softmax()]).predict(data)
        pred_prob_labels_full = model(data).numpy()
        pred_prob_labels = list(map(lambda l: l[0], pred_prob_labels_full))
    else:
        pred_prob_labels = npsum(model.predict_proba(data), axis=1)

    return list(pred_prob_labels)

