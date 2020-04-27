import pickle
from test_model.py import test_model


def build_model(name: str, all_layer_sizes, data_dir, epochs=10):
    """Build a linear neural net with a given name, save it in this directory.

    Args:
        name: the name that the NN will be saved as
        all_layer_sizes: a list of all layer sizes, from input to output.
            Therefore, the smallest size it can have is two. The size of the
            first layer should be the same as the length of each feature
            vector in the data sets
        data_dir: the directory in which to find the train, eval, and test
            pickled files
    """

    # Build the scaffolding
    linear_model = tf.keras.Sequential()
    linear_model.add(Dense(all_layer_sizes[1], input_dim=all_layer_sizes[0]))
    for layer_size in all_layer_sizes[2:]:
        linear_model.add(Dense(layer_size))

    # Obtain the datasets
    train_ds = pickle.load(data_dir + "/train").map(parse_example)
    eval_ds = pickle.load(data_dir + "/eval").map(parse_example)
    test_ds = pickle.load(data_dir + "/test").map(parse_example)

    # Train loop
    linear_model.fit(train_ds,
                     validation_data=eval_ds,
                     steps_per_epoch=len(train_ds)/epochs,
                     validation_steps=len(eval_ds)/epochs,
                     epochs=epochs,
                     verbose=True)

    # Show results of NN on test data set
    test_model(name, test_ds)


def parse_example(feature_list):
    """Parses each feature vector into a format this tensorflow NN can use."""

    return feature_list  # currently just one-to-one; I don't get the data types
    

