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

def remove_meta(file_names):
    # removing first & last 50 words
    texts = []
    for file_name in file_names:
        with open(file_name) as inp:
            text = inp.read().replace('\n', ' ')
            splitted = text.split(" ", 50)[50]
            tokenised = splitted.split(" ")
            ls = tokenised[:len(tokenised)-50]
            text = " ".join(ls)
            texts.append(text)
    return texts

def sample(doc, total_len, num_samples):
    tokenised = doc.split(" ")

    '''first_sample = tokenised[:30]
    last_sample = tokenised[-30:]
    middle_sample = tokenised[find_section(tokenised, 0.5)-20:find_section(tokenised, 0.5)+20]
    return " ".join(first_sample) + " " + " ".join(last_sample) + " " + " ".join(middle_sample)'''

    samples = []
    for i in range(num_samples):
        start = i * (total_len // num_samples)
        end = (i + 1) * (total_len // num_samples)
        sample = ' '.join(tokenised[start:end])
        sample = ' '.join(sample.strip().split())
        samples.append(sample)
    return ' '.join(samples)

def find_section(tokenised, prop):
    return int(len(tokenised) * prop)

def main():
    total_len = 100
    num_samples = 3
    file_names = get_files(book_dir)

    #Labels
    years = [int(os.path.basename(x)[:4]) for x in file_names]

    #Texts
    docs = remove_meta(file_names)
    for i, doc in enumerate(docs):
        docs[i] = sample(doc, total_len, num_samples)

    return docs, years

if __name__ == '__main__':
    main()
