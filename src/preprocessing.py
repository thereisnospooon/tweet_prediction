from gensim import models, corpora
from gensim.matutils import corpus2csc
import pandas as pd
import re
import logging
import numpy as np

TWEET_BEGIN = 2
TWEET_END = -2
TWEETS = 'tweet'
DIR = '../'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def process_data(X):
    """
    :param X: X is a pandas DataFrame of tweets (with no labels)
    :return: The feature matrix where each row is a tweet and each column is a feature. Ready to train/predict.
    """
    tweets = process_texts(X)
    dictionary = corpora.Dictionary.load('dictionary.dict')
    lsi_model = models.LsiModel.load('lsi.model')
    # Transform each tweet (a string) to a row of bag of words vector in the corpus matrix.
    corpus = [dictionary.doc2bow(tweet.split()) for tweet in tweets]
    # transform the bag of words corpus matrix to LSI matrix of features.
    # To read more about LSI - https://en.wikipedia.org/wiki/Latent_semantic_analysis
    feature_mat = corpus2csc(lsi_model[corpus]).T.toarray()
    return feature_mat


def clean_tweet(tweet):
    """
    Cleans each tweet from the punctuation marks in the re.sub call
    :param tweet: A tweet.
    :return:
    """
    tweet = tweet[TWEET_BEGIN:TWEET_END]  # Clean brackets and apostrophes
    tweet = tweet.lower()
    tweet = re.sub(r"(“|”|\”|—|\.|\,|:|\+|!|\?|\"|-|\(|\))", "", tweet)
    return tweet


def process_texts(texts):
    """
    :param texts: An iterable of strings.
    :return: An iterable of cleaned strings.
    """
    res = []
    for i in range(len(texts)):
        res.append(clean_tweet(texts[i]))
    return res

