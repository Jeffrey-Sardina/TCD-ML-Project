import glob
import os
import matplotlib.pyplot as plt

book_dir = 'clean_enough/'

def get_files(dir):
    return glob.glob(os.path.join(book_dir, '*.txt'))

def main():
    book_file_names = get_files(book_dir)
    lengths = []
    with open('lengths.csv', 'w') as out:
        for book_file_name in book_file_names:
            with open(book_file_name, 'r') as inp:
                num_words = 0
                for line in inp:
                    line = line.strip()
                    num_words += line.count(' ')
            print(num_words, file=out)

if __name__ == '__main__':
    main()
