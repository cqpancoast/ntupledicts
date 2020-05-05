import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def make_neuralnet(train_features, train_truth, validation_data=None, 
        hidden_layers=[], epochs=10):
    """TODO"""

    num_data = train_features.shape[0]
    feature_dim = train_features.shape[1]

    layer_sizes = iter([feature_dim] + hidden_layers + [2])  #TODO generalize past two

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
    linear_model.fit(train_features, train_truth,
                     validation_data=validation_data,
                     steps_per_epoch=steps_per_epoch,
                     epochs=epochs,
                     verbose=True)

    return linear_model


def test_model(model, test_features, test_truth):
    """Test a model's performance on a test dataset."""

    return model.evaluate(test_features, test_truth)

