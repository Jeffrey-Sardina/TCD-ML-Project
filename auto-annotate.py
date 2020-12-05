import glob
import os
from dateutil.parser import parse
from dateutil import parser
import re

book_dir = 'annotate_remove/'

def is_valid_date(date_str):
    try:
        parser.parse(date_str)
        return True
    except:
        return False

def remove500words(doc, n):
    tokenised = doc.split(" ")
    removed = tokenised[n:]
    return removed

def findDates(doc):
    return re.match('.*([1-3][0-9]{3})', doc)

def get_files(dir):
    return glob.glob(os.path.join(book_dir, '*.txt'))

def main():
    book_file_names = get_files(book_dir)
    # print(book_file_names)
    for book_file_name in book_file_names:
        with open(book_file_name, 'r') as inp:
            doc = inp.readlines()
            year = findDates(doc)
            directory = 'auto_annotated'
            if not os.path.exists(directory):
                os.makedirs(directory)
            dpath = '%s/%s.txt' % (directory, year)
            uniq = 1
            while os.path.exists(dpath):
                dpath = '%s/%s_%d.txt' % (directory, year, uniq)
                uniq += 1
            # with open(dpath, 'w') as out:
            print(dpath)


if __name__ == '__main__':
    main()
