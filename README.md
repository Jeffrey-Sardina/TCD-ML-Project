# Group 42: Life, the Universe, and Everything
## Claire Farrell and Jeffrey Sardina

## Repo structure

Directories
- annotated/
    - contains books whose year has been recorded
    - Files should be named xxxx.txt, where xxxx is the 4-digit year
    - Years used should for the data a book was finished being written English. If the book is a translation, it should be the year the book was translated to English.

Files
- .gitignote
    - Any file larger than 25MB should be ignored since Github will refuse to store it
    - Any large folders (such as the raw data) also should be ignored
- countwords.py
    - Counts the number of total words in all the raw data folder (not in this repo) and prints the counts out to lengths.csv
- preprocess.py
    - Removes some of the meta-data in a book. Not all meta-data is structured, so some books may still have some that should be removed later manually when years are annotated.
- sample.py
    - Takes a random samples to text from a file. All samples should be of the same length across books.
    - Not yet implemented
- workflow
    - Describers the workflow we have used to far to generate our data
