import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors

#Code from lectures (reformatted)
def ex_tokenization():
    #-----------------------------Tokenization
    nltk.download('punkt')
    tokenizer = CountVectorizer().build_tokenizer()
    print(tokenizer("Here's example text, isn’t it?"))
    print(WhitespaceTokenizer().tokenize("Here's example text, isn’t it?"))
    print(word_tokenize("Here's example text, isn’t it?"))
    print(tokenizer("likes liking liked"))
    print(WhitespaceTokenizer().tokenize("likes liking liked"))
    print(word_tokenize("likes liking liked"))
    stemmer = PorterStemmer()
    tokens = word_tokenize("Here's example text, isn’t it?")
    stems = [stemmer.stem(token)for token in tokens]
    print(stems)
    tokens = word_tokenize("likes liking liked")
    stems = [stemmer.stem(token)for token in tokens]
    print(stems)

def ex_bag_or_words():
    #-----------------------------bag of words
    docs = ['This is the first document.','This is the second second document.','And the third one.','Is this the first document?']
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)
    print(vectorizer.get_feature_names())
    print(X.toarray())

    vectorizer = CountVectorizer(ngram_range=(2, 2))
    X = vectorizer.fit_transform(docs)
    print(vectorizer.get_feature_names())
    print(X.toarray())

def ex_tf_idf():
    #-----------------------------TF-IDF
    docs = ['This is the first document.','This is the second second document.','And the third one.','Is this the first document?']
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)
    print(vectorizer.get_feature_names())
    print(X.toarray())

    vectorizer = TfidfVectorizer(norm=None)
    X = vectorizer.fit_transform(docs)
    print(vectorizer.get_feature_names())
    print(X.toarray())

def ex_recommender():
    #-----------------------------Recommending news articles example
    # first 1000 articles from news dataset at https://www.kaggle.com/snapcrack/all−the−news
    text = pd.read_csv('articles1_1000.csv')
    text.head()
    x = text['content']
    vectorizer = TfidfVectorizer(stop_words = 'english', max_df=0.2)
    X = vectorizer.fit_transform(x)
    print(x.size)
    print(X.size)

    indices = np.arange(x.size)
    train, test = train_test_split(indices, test_size=0.2)
    nbrs = NearestNeighbors(n_neighbors=3,metric=cosine_distances).fit(X[train])
    test=[test[0]]
    found = nbrs.kneighbors(X[test], return_distance=False)
    test_i=0
    print('text:\n%.300s'%x[test[test_i]])
    for i in found[0]:
        print('match %d:\n%.300s'%(i,x[train[i]]))
