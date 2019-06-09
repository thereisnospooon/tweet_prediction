import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from preprocessing import process_data

TWEETS = 'tweet'
LABEL = 'user'


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Read data
df = pd.read_csv('../train_data.csv', index_col=0)
df.reset_index(inplace=True)
dataset = process_data(df[TWEETS])
y = df[LABEL]

# Split data
X_train, X_test, y_train, y_test = train_test_split(dataset, y, train_size=0.8)

# Logistic Regression
estimator = LogisticRegression(n_jobs=4, verbose=True, multi_class='multinomial', solver='newton-cg')

estimator.fit(X_train, y_train)
y_hat = estimator.predict(X_test)
print("Accuracy of model: " + str(sum(y_hat == y_test) / len(y_test)))
