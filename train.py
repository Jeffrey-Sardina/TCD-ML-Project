import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#hyperparameters
#to tf-idf tokenizer
min_len = 2
max_len = 3
min_df = 0
max_df= 1

def create_input_array(documents):
    '''
    This function creates a bag of words (X / input) matrix to be used in the model based on the given documents
    '''
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(min_len, max_len), min_df=min_df, max_df=max_df)
    X = vectorizer.fit_transform(documents)
    print('bag of words sumary')
    print(vectorizer.get_feature_names())
    print(X.toarray())
    print()
    return X

def main():
    #Gather and process data into a form that we can run ML on
    documents, y = docs = ['This is the first document.','This is the second second document.','And the third one.','Is this the first document?'], [2020, 2020, 2020, 2020] #Call to sampling.py
    X = create_input_array(documents)

    #Split into train and test segments
    indices = np.arange(len(X))
    train, test = train_test_split(indices, test_size=0.2)

    #

if __name__ == '__main__':
    main()