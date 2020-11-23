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

def remove_meta(file):
    # removing first & last 50 words
    splitted = file.split(" ", 50)[50]
    tokenised = splitted.split(" ")
    ls = ls[:len(ls)-50]
    return " ".join(ls)

def sample(doc):
    tokenised = doc.split(" ")
    first_sample = tokenised[:30]
    last_sample = tokenised[-30:]
    middle_sample = tokenised[find_middle(tokenised)-20:find_middle(tokenised)+20]
    return " ".join(first_sample) + " " + " ".join(last_sample) + " " + " ".join(middle_sample)

def find_middle(tokenised):
    middle = float(len(tokenised))/2
    if middle % 2 != 0:
        return tokenised[int(middle - .5)]
    else:
        return (tokenised[int(middle)], tokenised[int(middle-1)])

def main():
    file_names = get_files(book_dir)
    doc = remove_meta(file_names)
    return sample(doc)

if __name__ == '__main__':
    main()
