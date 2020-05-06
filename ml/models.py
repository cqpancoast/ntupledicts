from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import GradientBoostingClassifier


def make_neuralnet(train_data, train_labels, validation_data=None,
                   hidden_layers=[], epochs=10):
    """Makes a neural net in tensorflow using training data and optional
    validation data.

    Args:
        train_data:  a tf tensor, indexed by track on the 0th axis and
            track property on the 1st axis, containing a track property
            value
        train_labels:  a tf tensor indexed by track, containing the
            trackset's labeled data
        validation_data:  a 2-tuple contianing a data/label pair in the
            same form as train_data and train_labels for use in validation
        hidden_layers:  a list of hidden layer sizes. By default, there are
            no hidden layers, just the input and the output, which are
            predetermined by data dimension
        epochs:  how many time the neural net is trained on data

    Returns:
        A trains tensorflow neural net.
    """

    num_data = train_data.shape[0]
    data_dim = train_data.shape[1]

    layer_sizes = iter(
        [data_dim] + hidden_layers + [2])  # TODO generalize

    # Build the scaffolding
    linear_model = Sequential()
    input_dim = next(layer_sizes)
    linear_model.add(Dense(next(layer_sizes), input_dim=input_dim))
    for layer_size in layer_sizes:
        linear_model.add(Dense(layer_size))

    # Compile
    linear_model.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])

    # Print summary
    linear_model.summary()

    # Train loop
    steps_per_epoch = num_data / epochs
    linear_model.fit(train_data, train_labels,
                     validation_data=validation_data,
                     steps_per_epoch=steps_per_epoch,
                     epochs=epochs,
                     verbose=True)

    return linear_model


def make_gbdt(train_data, train_labels):
    """Make a gradient boosted decision tree in sklearn using training
    data, using Claire's model as reference for creation parameters."""

    gbdt = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=23)
    gbdt.fit(train_data, train_labels)

    return gbdt
