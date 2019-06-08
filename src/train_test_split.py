import numpy as np
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

DIR = 'tweets_data/'
USER = 'user'
UP_FOLDER = '../'
TWEET = 'tweet'


def get_all_data():
    res = pd.DataFrame(columns=[USER, TWEET])
    directory = os.fsencode(UP_FOLDER + DIR)
    for file in os.listdir(directory):
        filename = UP_FOLDER + DIR + os.fsdecode(file)
        if filename.endswith(".csv") and not filename.endswith("demo.csv"):
            res = pd.concat([res, pd.read_csv(filename)])
            continue
    return res


def split_data(frame, train_size=0.8):
    y = frame[USER]
    frame = frame.drop(USER, axis=1)
    return train_test_split(frame, y, train_size=train_size, stratify=y)


if __name__ == '__main__':
    df = get_all_data()
    logging.info("Loaded DataFrame")
    X_train, X_test, y_train, y_test = split_data(df)
    X_train[USER] = y_train
    X_test[USER] = y_test
    logging.info("Data is split. Saving...")
    X_train.to_csv(UP_FOLDER + 'train_data.csv')
    X_test.to_csv(UP_FOLDER + 'test_data.csv')
