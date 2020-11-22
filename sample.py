import glob
import os

'''
An idea:

remove the first and last 50 words, ust in case we missed some meta-data, and sample from the rest

We need to either get documents of the same length only, or normalize for length. I like selecting fixed length sets, as otherwise the bag of words will be far too large.

100 words then remove stopwords

'''

book_dir = 'annotated/'

def get_files(dir):
    return glob.glob(os.path.join(book_dir, '*.txt'))

def remove_trailing(file):
    # removing first & last 50 words
    pass



def main():
    file_names = get_files(book_dir)

if __name__ == '__main__':
    main()
