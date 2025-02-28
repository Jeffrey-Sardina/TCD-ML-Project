# Group 42: Life, the Universe, and Everything
## Claire Farrell and Jeffrey Sardina

## Repo structure

Directories
- annotated/
    - contains books whose year has been recorded. Also nmay be used as a staging area for a small set of books you intend to annotate.
    - Files should be named xxxx.txt, where xxxx is the 4-digit year.
    - Years used should for the data a book was finished being written English. If the book is a translation, it should be the year the book was translated to English. Publication dates (but NOT dates a book was released on Project Gutenberg) may be used to approximate when a book was written.
    - Transcriber's notes, translator's notes, tables of contents, 'illustrated by's and other text not added by Project Gutenberg should also be removed. We only want to keep the raw text of the book. I'm even removing the author's name, since that may be used to over-fit / bias the AI against learning the text of the book. Many of these notes contain the date the book was written--which helps us, but is data the algorithms should not be able to see.
    - In the case multiple books are made in the same year, differentiate them as xxx.t5xt, xxxx_2.txt, etc.
- annotated_remove/
    - contains books to be annotated. Please delete them from this folder once you have annotated it

Files
- .gitignore
    - Any file larger than 25MB should be ignored since Github will refuse to store it.
    - Any large folders (such as the raw data) also should be ignored.
- countwords.py
    - Counts the number of total words in all the given data folder (not in this repo) and prints the counts out to a csv.
- lengths_annotated.csv
    - contains the lengths of all books in the annotated/ folder
- lengths.csv
    - contains the lengths of all books in the clean_enough/ folder
- preprocess.py
    - Removes some of the meta-data in a book. Not all meta-data is structured, so some books may still have some that should be removed later manually when years are annotated.
- sample.py
    - Takes a random samples to text from a file. All samples should be of the same length across books.
    - Not yet implemented.
- train.py
    - For code to be used in training our model.
    - Not yet implemented.
- workflow
    - Describers the workflow we have used to far to generate our data.
