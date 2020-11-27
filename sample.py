import glob
import os

book_dir = 'annotated/'

def get_files(folder):
    return glob.glob(os.path.join(folder, '*.txt'))

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
    total_len = 2000
    num_samples = 20
    run_sample(total_len, num_samples)

def run_sample(total_len, num_samples):
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
