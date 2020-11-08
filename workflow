Linux: (download all English books)
    (run 07-11-2020) (ref https://webapps.stackexchange.com/questions/12311/how-to-download-all-english-books-from-gutenberg)
    wget -H -w 2 -m http://www.gutenberg.org/robot/harvest?filetypes[]=txt&langs[]=en \
    --referer="http://www.google.com" \
    --user-agent="Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.6) Gecko/20070725 Firefox/2.0.0.6" \
    --header="Accept: text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png,*/*;q=0.5" \
    --header="Accept-Language: en-us,en;q=0.5" \
    --header="Accept-Encoding: gzip,deflate" \
    --header="Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7" \
    --header="Keep-Alive: 300"

PS: (unzip all into save dir)
    (https://social.technet.microsoft.com/Forums/en-US/ee897671-fa3a-4e06-8cb6-0986900c032e/powershell-function-unzip-all-files-in-a-folder)
    Get-ChildItem 'path to folder' -Filter *.zip | Expand-Archive -DestinationPath 'path to extract' -Force

Filesystem:
    Copy out all .txt files

Linux: (convert all to utf8)
    (http://www.f15ijp.com/2010/02/linux-converting-a-file-encoded-in-iso-8859-1-to-utf-8/)
    find . -name "*.txt" -exec iconv -f ISO-8859-1 -t UTF-8 {} -o conv/{}.txt \;

Python:
    Run preprocess.py

Filesystem:
    RM some corrupted files (non-English)
    RM empty files

Linux: (remove duplicates)
    rm *-8.txt
    rm *-0.txt

Count words in cleaned files
    Run count_words.py

Todo
    samples files and label by year
