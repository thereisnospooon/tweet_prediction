from gensim import models, corpora
from preprocessing import process_texts
import pandas as pd
from gensim.models.phrases import Phraser, Phrases
from six import iteritems
import logging
from nltk.corpus import stopwords

DIR = '../'
TWEETS = 'tweet'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def create_dictionary(texts, dest_file: str, build_bigram, working_directory=DIR):
    """
    Reads the file specified by source_file, creates a dictionary and saves it to the dest_file
    path.
    :param working_directory: The path to the directory where the bigram model files should be saved.
    :param build_bigram: 1 if building a new phrases object is needed else an already processed bigram model will
                         be loaded.
    :param source_file: path to source text file.
    :param dest_file: path to save dictionary to.
    :return:
    """
    # collect statistics about all tokens
    stoplist = stopwords.words('english')
    if build_bigram:
        bigram = Phrases([tweet.split() for tweet in texts])
        bigram.save(working_directory + '/bigram_model.phrase')
    else:
        bigram = Phrases.load(working_directory + '/bigram_model.phrase')
    phraser = Phraser(bigram)
    # Build dictionary
    dictionary = corpora.Dictionary(phraser[line.lower().split()] for line in texts)
    # remove stop words and words that appear only once
    stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
                if stopword in dictionary.token2id]
    once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
    dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
    dictionary.filter_extremes(no_below=0.3, no_above=0.85)
    dictionary.compactify()  # remove gaps in id sequence after words that were removed
    dictionary.save(dest_file)
    print(dictionary)
    print(dictionary.token2id)
    return dictionary


if __name__ == '__main__':
    df = pd.read_csv(DIR + 'train_data.csv', index_col=0)
    df.reset_index(inplace=True)
    tweets = df[TWEETS]
    tweets = process_texts(tweets)
    dictionary = create_dictionary(tweets, DIR + 'dictionary.dict', True)
    corpus = [dictionary.doc2bow(tweet.split()) for tweet in tweets]
    lsi = models.LsiModel(corpus=corpus, id2word=dictionary, num_topics=200)
    lsi.save(DIR + 'lsi.model')
