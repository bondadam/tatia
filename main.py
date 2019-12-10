import nltk
import os
import re
from collections import Counter



def getTexts():
    """
    Lis les les deux textes présents dans les fichier
    les mets dans des strings qu'il rend
    puis close les fichiers
    """
    firstText = ""
    secondText = ""
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    textFiles = []
    for f in files:
        if f[-4:] == ".txt":
            textFiles.append(f)
    if len(textFiles) > 2 or len(textFiles) < 2:
        raise Exception("Erreur: vérifiez qu'il y a bien 2 fichiers textes à comparer.")
    with  open(textFiles[0], 'r') as f1, open(textFiles[1], 'r') as f2:
        firstText = f1.read()
        secondText = f2.read()
    return [firstText, secondText]


def unpunctuate(givenText):
    """Strip punctuation then return words as array"""
    nonPunct = re.compile('.*[A-Za-z0-9].*')
    filtered = [w for w in givenText.split() if nonPunct.match(w)]
    return filtered


def countRealWords(givenText):
    """return word count in text"""
    counts = Counter(givenText)
    return counts
    
    

def main():
    texts = getTexts()
    
    print("%d words in text 1, %d words in text 2" % (len(unpunctuate(texts[0])), len(unpunctuate(texts[1]))))
    #print(countRealWords(unpunctuate(texts[0])))
  
if __name__== "__main__":
  main()
