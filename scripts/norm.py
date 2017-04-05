import sys
import re
import string

def normalize(text):
    text = text.replace("\n","")
    text = text.replace("\s+"," ")    
    text = re.sub(r"http\S+", "#URL#",text)
    text = re.sub(r"@\w+", "#RETWEET#",text)
    text = re.sub(r"[^#\w\s]+","",text)
    text = re.sub(r"\d+","###",text)
    text = re.sub(r"\r","",text)
    
    return text

labels = {}
with open(sys.argv[1]) as fh:
    for line in fh:
        line = line.strip()
        (label, text) = line.split(' ', 1)
        print("{} {}".format(label,normalize(text)))
        
