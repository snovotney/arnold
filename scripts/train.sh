#!/bin/bash

# Assumed input is text data with one tweet per line, where first word is the label
# Author: Scott Novotney, novotney@isi.edu

if [ $# != 1 ]; then
    echo "Must pass in training file, one tweet per line, first word being the label"
    exit 1
fi

myshuf() {
  perl -MList::Util=shuffle -e 'print shuffle(<>);' "$@";
}

check_file() {
    file=$1;
    if [ -e $file ]; then
        while true; do
            echo "File $1 exists"
            read -p "Do you want to replace it? [yes/no] " yn
            case $yn in
                [Yy]* ) return 0;;
                [Nn]* ) return 1;;
                * ) echo "Please answer yes or no.";;
            esac
        done
    else
        return 0
    fi
}

set -e
set -u

export WRKDIR="$( cd "$( dirname "${BASH_SOURCE[0]}";  )"/.. && pwd )"
RESULTSDIR=$WRKDIR/results
mkdir -p $RESULTSDIR

# Get the basename
rawdata=$1
filename="$( basename $rawdata )"
extension="${filename##*.}"
filename="${filename%.*}"
data="$RESULTSDIR/$filename"

# Run through normalization and Farasa pre-processor
if check_file "$data.csv"; then

    cat $rawdata | myshuf > $data.shuf
    # remove the labels from normalization
    cut -d' ' -f 2- $data.shuf > $data.txt
    cut -d' ' -f 1 $data.shuf > $data.lbls

    # map retweets, urls and numbers to one token, deal with whitespace stuff
    python3 $WRKDIR/scripts/norm.py $data.txt > $data.norm

    # Run Farasa morpheme splitter
    java -jar $WRKDIR/bin/FarasaSegmenterJar.jar --input $data.norm --output $data.farasa

    paste $data.lbls $data.farasa -d',' > $data.csv

    rm $data.farasa $data.txt $data.lbls $data.norm $data.shuf
fi

# Train logistic regression 
python3 scripts/train_logit.py $data.csv $RESULTSDIR/model.pkl
