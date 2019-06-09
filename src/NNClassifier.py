import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from preprocessing import process_data
from keras.models import Sequential
from keras import regularizers
from keras import initializers
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

TWEETS = 'tweet'
LABEL = 'user'

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# define baseline model
# Current architecture -
# 200 dimension input -> [10 neurons hidden layer] -> 10 classes ouput
# Todo: You can play with the architecture as you like :)
def baseline_model():
    # create model
    model = Sequential()
    # TODO: Make sure input_dim here matches shape[1] of feature_mat from process data
    # model.add(Conv1D(32, kernel_size=2,
    #                  activation='relu', input_shape=(208, 1)))
    model.add(Dense(40, input_dim=num_features,
                    activation='relu',
                    # kernel_regularizer=regularizers.l2(),
                    kernel_initializer=initializers.glorot_normal()
                    ))
    model.add(Dense(200,
                    activation='relu',
                    # kernel_regularizer=regularizers.l2(),
                    kernel_initializer=initializers.glorot_normal()
                    ))
    # model.add(Dense(20,
    #                 activation='relu',
    #                 # kernel_regularizer=regularizers.l2(),
    #                 kernel_initializer=initializers.glorot_normal()
    #                 ))
    # model.add(Dense(20,
    #                 activation='relu',
    #                 # kernel_regularizer=regularizers.l2(),
    #                 kernel_initializer=initializers.glorot_normal()
    #                 ))
    # model.add(Dense(20,
    #                 activation='relu',
    #                 # kernel_regularizer=regularizers.l2(),
    #                 kernel_initializer=initializers.glorot_normal()
    #                 ))
    # model.add(Dense(20,
    #                 activation='relu',
    #                 # kernel_regularizer=regularizers.l2(),
    #                 kernel_initializer=initializers.glorot_normal()
    #                 ))
    # model.add(Dense(10,
    #                 activation='relu',
    #                 # kernel_regularizer=regularizers.l2(),
    #                 kernel_initializer=initializers.glorot_normal()
    #                 ))
    # model.add(Conv1D(32, kernel_size=3,
    #                  activation='relu'))
    # model.add(Flatten())
    model.add(Dense(10, activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Read data
df = pd.read_csv('../train_data.csv', index_col=0)
df.reset_index(inplace=True)
dataset, num_features = process_data(df[TWEETS])
# Line below for convulutional nets
# dataset = dataset.reshape((len(dataset), num_features, 1))
y = df[LABEL]

# Encode y to OHE
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
dummy_y = np_utils.to_categorical(encoded_y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(dataset, dummy_y, train_size=0.8)

# Neural Network
estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=5, verbose=1)

# Fit, and estimate performance.
estimator.fit(X_train, y_train)
y_hat = estimator.predict(X_test)
y_orig = np.argmax(y_test, axis=1, out=None)
print("Accuracy of model: " + str(sum(y_hat == y_orig) / len(y_orig)))
