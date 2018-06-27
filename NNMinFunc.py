import numpy as np
from keras import Sequential,initializers
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
import matplotlib.pyplot as plt

def create_dataset(m = 900000):
    """
    Generates data set
    m: number of data entries
    """
    # Generates random numbers
    x = np.random.rand(m, n_items)

    # Rounding the data
    x = np.around(x, decimals=2)

    # Find the minimum of x
    y = np.argmin(x, axis=1)

    # One hot encoding
    y = np.eye(n_items)[y]

    return x, y

def build_model():
    """
    Creating an ANN model
    :return: model
    """
    # The model is fully connected model
    model = Sequential()

    init = initializers.RandomNormal(mean=0.5, stddev=0.5, seed=1234)

    # Add first hidden layer to the model the Bias is selected between 0 to 1 in order to normalize
    model.add(Dense(n_items, input_dim=n_items, activation='sigmoid',bias_initializer=init))
    # Add second hidden layer
    model.add(Dense(n_items, activation='sigmoid', bias_initializer=init))
    # Output layer
    model.add(Dense(n_items, activation='sigmoid', bias_initializer=init))
    # Using Adam instead of SGD, the learning rate is selected by trail and error
    optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999)
    # The loss function is binary crossentropy because we want each neuron of output be 0 or 1
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model
def train_model():
    # Create data set
    x_train, y_train = create_dataset()
    # Split data set to train and dev sets (we want to be sure that dev set is not inside the train set)
    x_train = x_train[:-1000]
    y_train = y_train[:-1000]
    x_val = x_train[-1000:]
    y_val = y_train[-1000:]
    # Build the model
    model = build_model()
    # Train the model
    model.fit(x_train, y_train, epochs=40, batch_size=32)
    # Save the model
    model.save('./models/25inputs_model.h5', overwrite=True)
    # Evaluate model against dev test
    loss, accuracy = model.evaluate(x_val, y_val)
    print(f'Loss = {loss}, accuracy = {accuracy}')
    # Test the model
    X_test, _ = create_dataset(10)
    for inp in X_test:
        print(np.min(inp), inp[np.argmax(model.predict(np.expand_dims(inp, axis=0)))])


if __name__ == '__main__':
    n_items = 25

    loaded_model = load_model('./models/25inputs_model.h5')
    X_test, _ = create_dataset(5)
    for inp in X_test:
        print(np.min(inp), loaded_model.predict(np.expand_dims(inp, axis=0)))
