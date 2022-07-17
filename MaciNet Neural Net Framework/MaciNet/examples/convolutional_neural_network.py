
from __future__ import print_function
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

from MaciNet.deep_learning import NeuralNetwork
from MaciNet.utils import train_test_split, to_categorical, normalize
from MaciNet.utils import get_random_subsets, shuffle_data, Plot
from MaciNet.deep_learning.optimizers import StochasticGradientDescent, Adam, RMSprop, Adagrad, Adadelta
from MaciNet.deep_learning.loss_functions import CrossEntropy
from MaciNet.deep_learning.layers import Dense, Dropout, Conv2D, Flatten, Activation, MaxPooling2D
from MaciNet.deep_learning.layers import AveragePooling2D, ZeroPadding2D, BatchNormalization, RNN



def main():

    filename: str = 'SotechDataset_Sensors12.csv'

    # read each value, check for missing / erroneous data, and import the data into a pandas dataframe
    dataframe: pd.DataFrame = pd.read_csv(filename, header=None)
    y_vals_nn = dataframe[16]
    x_vals_nn = dataframe.drop([16], 1)

    # now declare these vals as np array
    # this ensures they are all of identical data type and math-operable
    x_data_train = np.array(x_vals_nn)
    y_data_train = np.array(y_vals_nn)

    X = np.expand_dims(x_data_train, axis=0)
    y = y_data_train

    # Convert to one-hot encoding
    y = to_categorical(y.astype("int"))
    n_samples, n_features, depth = X.shape
    n_hidden = n_features

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, seed=37)

    model = NeuralNetwork(optimizer=Adam(),
                        loss=CrossEntropy,
                        validation_data=(X_test, y_test))

    model.add(Conv2D(n_filters=16, filter_shape=(3,3), stride=1, input_shape=(1, n_hidden, depth), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Conv2D(n_filters=32, filter_shape=(3,3), stride=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(y.shape[1]))
    model.add(Activation('softmax'))
    model.summary(name="ConvNet")

    train_err, val_err = model.fit(X_train, y_train, n_epochs=100, batch_size=32)

    # Training and validation error plot
    n = len(train_err)
    training, = plt.plot(range(n), train_err, label="Training Error")
    validation, = plt.plot(range(n), val_err, label="Validation Error")
    plt.legend(handles=[training, validation])
    plt.title("Error Plot")
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.show()

    _, accuracy = model.test_on_batch(X_test, y_test)
    print ("Accuracy:", accuracy)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    X_test = X_test.reshape(-1, 8*8)
    # Reduce dimension to 2D using PCA and plot the results
    Plot().plot_in_2d(X_test, y_pred, title="Convolutional Neural Network", accuracy=accuracy, legend_labels=range(10))

if __name__ == "__main__":
    main()

