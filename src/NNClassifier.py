import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from preprocessing import process_data
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

TWEETS = 'tweet'
LABEL = 'user'

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# define baseline model
# Current architecture -
# 200 dimension input -> [10 neurons hidden layer] -> 10 classes ouput
# Todo: You can play with the architecture as you like. just make sure that the output of each layer (the first argument
# Todo: to the "Dense" function matches the input_dim parameter of the subsequent layer :)
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(100, input_dim=200, activation='relu'))
    model.add(Dense(10, input_dim=100, activation='relu'))
    # model.add(Dense(20, input_dim=50, activation='relu'))
    # model.add(Dense(20, input_dim=10, activation='relu'))
    # model.add(Dense(50, input_dim=20, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Read data
df = pd.read_csv('../train_data.csv', index_col=0)
df.reset_index(inplace=True)
dataset = process_data(df[TWEETS])
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

estimator.fit(X_train, y_train)
y_hat = estimator.predict(X_test)
y_orig = np.argmax(y_test, axis=1, out=None)
print("Accuracy of model: " + str(sum(y_hat == y_orig) / len(y_orig)))
