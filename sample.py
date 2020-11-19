import glob
import os

book_dir = 'annotated/'

def get_files(dir):
    return glob.glob(os.path.join(book_dir, '*.txt'))

def main():
    file_names = get_files(book_dir)

if __name__ == '__main__':
    main()
