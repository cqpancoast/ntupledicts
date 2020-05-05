import pickle
import tensorflow as tf
import tensorflow.keras as keras


def make_neuralnet(train_features, train_truth, validation_data=None, 
        hidden_layers=[], epochs=10):
    """TODO"""

    # Build the scaffolding
    linear_model = tf.keras.Sequential()
    linear_model.add(Dense(all_layer_sizes[1], input_dim=all_layer_sizes[0]))
    for layer_size in all_layer_sizes[2:]:
        linear_model.add(Dense(layer_size))

    # Train loop
    steps_per_epoch = np.size(train_truth)
    linear_model.fit(train_features, train_truth,
                     validation_data=validation_data,
                     steps_per_epoch=steps_per_epoch,
                     epochs=epochs,
                     verbose=True)

    return linear_model


def test_model(model, test_features, test_truth):
    """Test a model's performance on a test dataset."""

    

