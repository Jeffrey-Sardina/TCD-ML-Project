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

def removeWords(doc, n, m):
    tokenised = doc.split(" ")
    removed = tokenised[n:][:-m]
    return ' '.join(removed)

def findDates(doc):
    string = re.search('([Cc][Oo][Pp][Yy][Rr][Ii][Gg][Hh][Tt].*1[6-9]{3})', doc).group()
    return re.search('(1[6-9]{3})', string).group()

def get_files(dir):
    return glob.glob(os.path.join(book_dir, '*.txt'))

def main():
    book_file_names = get_files(book_dir)
    count=0
    # print(book_file_names)
    
    for book_file_name in book_file_names:
        with open(book_file_name, 'r') as inp:
            try:
                doc = inp.read()
                year = findDates(doc)
                directory = 'auto_annotated/'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                dpath = '%s/%s.txt' % (directory, year)
                uniq = 1
                while os.path.exists(dpath):
                    dpath = '%s/%s_%d.txt' % (directory, year, uniq)
                    uniq += 1
                with open(dpath, 'w') as out:
                    doc1 = removeWords(doc, 1000, 50)
                    out.write(doc1)
                count+=1
                if count % 50 == 0:
                    print(count)
            except:
                print('nah')
    print(count)

if __name__ == '__main__':
    main()
