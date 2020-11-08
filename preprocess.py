import glob
import os

book_dir = 'utf8/'
out_dir = 'processed/'

def get_files(dir):
    return glob.glob(os.path.join(book_dir, '*.txt'))

def preprocess(book_file_name):
    base_name = os.path.basename(book_file_name)
    book_text_started = False
    title = ''
    lines = []

    with open(book_file_name, 'r') as inp:
        num_words = 0
        for line in inp:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith('*** START OF THIS PROJECT GUTENBERG EBOOK'):
                book_text_started = True
                title = line.replace('*** START OF THIS PROJECT GUTENBERG EBOOK ', '').replace(' ***', '')
                continue
            if line.startswith('*** END OF THIS PROJECT GUTENBERG EBOOK'):
                break
            if book_text_started:
                num_words += line.count(' ') + 1
                lines.append(line)

    if len(title) > 0:
        out_base_name = title + '_' + base_name
        out_file_name = os.path.join(out_dir, out_base_name)
        with open(out_file_name, 'w') as out:
            for line in lines:
                print(line, file=out)
    else:
        raise ValueError('file could not be read')

def main():
    book_file_names = get_files(book_dir)
    errors = 0
    for book_file_name in book_file_names:
        print(book_file_name)
        try:
            preprocess(book_file_name)
        except:
            errors += 1
            print('ERROR' + str(errors))

if __name__ == '__main__':
    main()

'''
*** START OF THIS PROJECT GUTENBERG EBOOK PARADISE LOST ***
...text...
*** END OF THIS PROJECT GUTENBERG EBOOK PARADISE LOST ***
'''
