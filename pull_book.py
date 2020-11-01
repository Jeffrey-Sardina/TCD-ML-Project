from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
import sys

def main():
    try:
        book_id = int(sys.argv[1])
    except:
        print('Usage: python pull_bokk.py <book_id>')
        exit(1)

    text = strip_headers(load_etext(book_id)).strip()
    with open(str(book_id) + '.full', 'w', encoding="utf-8") as out:
        for line in text.split('\n'):
            if len(line.strip()) > 0:
                out.write(line)

if __name__ == '__main__':
    main()