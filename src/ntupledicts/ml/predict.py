"""PREDICT: make and measure predictions on discriminator behavior."""

from .. import operations as ndops
from .. import analyze as ndanl
from ..operations import select as sel


def check_pred_labels_size(labels, pred_labels):
    """Raises a ValueError is the predicted labels are a different size
    than the actual labels. Otherwise, does nothing."""

    if len(labels) != len(pred_labels):
        raise ValueError("Predicted labels size differs from labels size")


def apply_threshold(pred_prob_labels, threshold):
    """Sends every prediction in the list below the threshold
    (exclusive) to zero and everything above it (inclusive) to one.
    In the parlance of this file, turns predicted probablistic labels
    into predicted labels. Returns a list."""

    return list(map(lambda pred: 1 if pred >= threshold else 0,
        pred_prob_labels))


def pred_proportion_given_truth_case(labels, pred_labels,
        label_restriction, pred_label_case, threshold=.6):
    """Look at the relative proportion of a value of the predicted
    probability labels, looking only at values who match to an acutal
    label of a particular case.

    This is the generalization of true and false positive rates.

    Args:
        labels: a list or tensor of binary classifer labels.
        pred_labels: a list of predicted binary classifier labels,
            OR a list of probablistic predictions to be converted to
            exact predictions using the threshold.
        label_restriction: a function returning true for values of the
            label case to which the domain should be restricted.
        pred_label_case: a function returning true for values to count
            part of the proportion in the prediction with restricted
            domain.
        threshold: a threshold to apply to the probablistic data
            before computing the agreement. Assumes binary classifier.

    Returns:
        The proportion of predicted values meeting a certain case given
        a restriction of true values meeting a certain case, and the
        error in prediction.

    Raises:
        ValueError: if the true and predicted labels differ in size.
    """

    check_pred_labels_size(labels, pred_labels)

    labels_meet_restriction = list(map(lambda label:
        bool(label_restriction(label)),
        labels if isinstance(labels, list) else labels.numpy()))
    pred_labels_meet_case = list(map(pred_label_case,
        apply_threshold(pred_labels, threshold)))

    domain_size = sum(labels_meet_restriction)
    if domain_size == 0:
        return 0, 0

    num_pred_labels_meet_case_in_domain = sum(map(
        lambda label_meets_restriction, pred_label_meets_case:
        label_meets_restriction and pred_label_meets_case,
        labels_meet_restriction, pred_labels_meet_case))

    return num_pred_labels_meet_case_in_domain / domain_size, ndanl.pred_error(
            domain_size, num_pred_labels_meet_case_in_domain)


def true_positive_rate(labels, pred_labels, threshold=.6):
    """For a binary classifier label, returns the proportion of "true"
    cases that the model predicted correctly. Throws an error if the
    lists are of different sizes.

    Args:
        labels: a list of binary classifier labels.
        pred_labels: a list of predicted binary classifier labels,
            OR a list of probablistic predictions to be converted to
            exact predictions using the threshold.
        threshold: a threshold to apply to the probablistic data.

    Returns:
        The proportion of "true" cases that a model predicted correctly
        and the prediction error.

    Raises:
        ValueError: if the true and predicted labels differ in size.
    """

    return pred_proportion_given_truth_case(labels, pred_labels,
            sel(1), sel(1), threshold)


def false_positive_rate(labels, pred_labels, threshold=.6):
    """For a binary classifier label, returns the proportion of "false"
    cases that the model predicted were "true". Raises an error if the
    lists are of different sizes.

    Args:
        labels: a list of binary classifier labels.
        pred_labels: a list of predicted binary classifier labels,
            OR a list of probablistic predictions to be converted to
            exact predictions using the threshold.
        threshold: a threshold to apply to the probablistic data.

    Returns:
        The proportion of "false" cases that a model predicted "true"
        and the error in prediction.

    Raises:
        ValueError: if the true and predicted labels differ in size.
    """

    return pred_proportion_given_truth_case(labels, pred_labels,
            sel(0), sel(1), threshold)


def predict_labels(model, data):
    """Run the model on each element of a dataset and produce a list of
    probabilistic predictions (note: not logits). Assumes a binary
    classifier. Does not apply a threshold.

    Args:
        model: a tensorflow or sklearn model capable of prediction.
        data: an array of elements that the model can use to make
            predictions.

    Returns:
        A Python list of probabilistic predictions.
    """

    # Different models predict in different ways
    if "keras" in str(type(model)):
        pred_prob_labels = map(lambda l: l[0], model.predict(data))
    else:
        pred_prob_labels = map(lambda l: l[1], model.predict_proba(data))

    return list(pred_prob_labels)


def predict_labels_cuts(tpd_selector, dataset):
    """Return a list of labels corresponding to which tracks were
    selected in a dataset. Interpreted in the context of this ML
    package as predicting some binary track property based on cuts.

    Args:
        tpd_selector: a selector for a track properties dict.
        dataset: a TrackPropertiesDataset.

    Returns:
        A Python list of labels corresponding to which tracks were
        selected in a dataset.
    """

    track_prop_dict = dataset.to_track_prop_dict()

    cut_indices = ndops.select_indices(track_prop_dict, tpd_selector)

    pred_labels = [1 for _ in range(
        ndops.track_prop_dict_length(track_prop_dict))]
    for cut_index in cut_indices:
        pred_labels[cut_index] = 0

    return pred_labels

