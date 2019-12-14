import nltk
import os
#import matplotlib
from scipy import stats ##requires numpy + mkl
from scipy import array
from collections import Counter

##NLTK packages to install : punkt

def getTexts():
    """
    Reads text files present in current directory, stores them in strings,
    closes the files then returns strings. Fails if >2 or <2 text files.
    """
    firstText = ""
    secondText = ""
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    textFiles = []
    for f in files:
        if f[-4:] == ".txt":
            textFiles.append(f)
    if len(textFiles) > 2:
        raise Exception("Erreur: >2 fichiers textes dans le dossier courant.")
    elif len(textFiles) < 2:
        raise Exception("Erreur: <2 fichiers textes dans le dossier courant.")
    with open(textFiles[0], 'r') as f1, open(textFiles[1], 'r') as f2:
        firstText = f1.read()
        secondText = f2.read()
    return [firstText, secondText]

def unpunctuate(tokenizedText):
    """Input: tokenized text
    Removes tokens that are only punctuation from array."""
    unpunctuatedText = []
    for token in tokenizedText:
        if any(character.isalpha() for character in token):
            unpunctuatedText.append(token)
    return unpunctuatedText

def wordFrequency(tokenizedText):
    """Input: tokenized text
    returns nb of times each word is in text"""
    counts = Counter(tokenizedText)
    return counts

def tokenize(textString):
    """Input: text string
    returns array of tokenized words"""
    return nltk.word_tokenize(textString, language='french')

def getTokenFreqDist(tokenizedText):
    """Input: tokenized text
    returns array of token length frequency distribution"""
    tokenLengths = [len(token) for token in tokenizedText]
    return nltk.FreqDist(tokenLengths)

def sortFreqDist(freqDist):
    return dict(sorted(freqDist.items()))

def compareTokenFreqDists(freqDist1, freqDist2):
    pass

def main():
    rawTexts = getTexts()
    tokenizedTexts = list(map(tokenize, rawTexts))
    unpunctuatedTexts = list(map(unpunctuate, tokenizedTexts))

    tokenFreqDists = list(map(getTokenFreqDist, unpunctuatedTexts))
    sortedFreqDists = list(map(sortFreqDist, tokenFreqDists))
    
    print("%d words in text 1, %d words in text 2" % (len(unpunctuatedTexts[0]), len(unpunctuatedTexts[1])))
    print(wordFrequency(unpunctuatedTexts[0]))

    #tokenFreqDists[0].plot(15,title="Frequency Distribution of text 1")
    #sortedFreqDists[0].plot(15,title="Frequency Distribution of text 1")

    print(tokenFreqDists[0].values())
    print(sortedFreqDists[0].values(), sortedFreqDists[1].values())

    array1 = array(list(sortedFreqDists[0].values()))
    array2 = array(list(sortedFreqDists[1].values()))
    #  Kolmogorov-Smirnov
    print("Result of Kolmogorov-Smirnov:\n", stats.ks_2samp(array1, array2))
  
if __name__== "__main__":
  main()
