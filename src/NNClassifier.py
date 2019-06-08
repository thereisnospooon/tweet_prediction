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
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(100, input_dim=200, activation='relu'))
    model.add(Dense(50, input_dim=100, activation='relu'))
    # model.add(Dense(20, input_dim=50, activation='relu'))
    # model.add(Dense(20, input_dim=10, activation='relu'))
    # model.add(Dense(50, input_dim=20, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


df = pd.read_csv('../train_data.csv', index_col=0)
df.reset_index(inplace=True)
dataset = process_data(df[TWEETS])
y = df[LABEL]

X_train, X_test, y_train, y_test = train_test_split(dataset, y, train_size=0.8)
# Encode y to OHE
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
dummy_y = np_utils.to_categorical(encoded_y)

# Neural Network
estimator = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=5, verbose=1)

estimator.fit(X_train, y_train)
y_hat = estimator.predict(X_test)
print(sum(y_hat == y_test) / len(y_hat))



