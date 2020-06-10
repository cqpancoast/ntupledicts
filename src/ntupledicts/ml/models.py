from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import GradientBoostingClassifier


def make_neuralnet(train_dataset, eval_dataset=None,
                   hidden_layers=[], epochs=10, classifier_order=1):
    """Makes a neural net in tensorflow using training data and optional
    validation data. Takes in the dimension of the data, has the number
    of specified hidden layers, and ends with probablistic output.

    Args:
        train_dataset: a TrackPropertiesDataset that the model will
            train on.
        eval_dataset: a TrackPropertiesDataset that the model will
            use to evalutate performance at the end of each epoch.
        hidden_layers: a list of hidden layer sizes. By default, there
            are no hidden layers, just the input and the output, which
            are predetermined by data dimension.
        epochs: how many time the neural net is trained on data.
        classifier_order: order of categorization. This is set to 1 at
            default, assuming a binary classifier.

    Returns:
        A trained tensorflow neural net.
    """

    # Build the scaffolding
    linear_model = Sequential()
    hidden_layer_sizes = iter(hidden_layers)
    linear_model.add(Dense(next(hidden_layer_sizes),
                           input_dim=train_dataset.get_data_dim()))
    for layer_size in hidden_layer_sizes:
        linear_model.add(Dense(layer_size, activation="relu"))
    linear_model.add(Dense(classifier_order, activation="sigmoid"))

    # Compile
    linear_model.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])

    # Train loop
    steps_per_epoch = train_dataset.size() / epochs
    validation_data = None if eval_dataset is None \
        else (eval_dataset.get_data(), eval_dataset.get_labels())
    linear_model.fit(train_dataset.get_data(), train_dataset.get_labels(),
                     validation_data=validation_data,
                     steps_per_epoch=steps_per_epoch,
                     epochs=epochs,
                     verbose=True)

    return linear_model


def make_gbdt(train_dataset, n_estimators=100, max_depth=3, random_state=23):
    """Make a gradient boosted decision tree in sklearn using training
    data, using Claire's model as reference for creation parameters.

    Args:
        train_dataset: a TrackPropertiesDataset that the model will
            train on.
        n_estimators, max_depth, random_state: check out the sklearn
            documentation for a GradientBoostingClassifier.

    Returns:
        A trained sklearn gradient boosted decision tree.
    """

    gbdt = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state)
    gbdt.fit(train_dataset.get_data().numpy(),
             train_dataset.get_labels().numpy())

    return gbdt
