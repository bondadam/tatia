import nltk
import os
import statistics
#import matplotlib
from scipy import stats ##requires numpy + mkl
from scipy import array
from collections import Counter
from nltk.corpus import stopwords
from tabulate import tabulate

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


def getFilenames():
    """Gets filenames of text files for display purposes"""
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    textFiles = []
    for f in files:
        if f[-4:] == ".txt":
            textFiles.append(f)
    return textFiles
    
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

def getStopWordFrequency(tokenizedText):
    stopWords = set(stopwords.words('french'))
    filteredArray = [word for word in tokenizedText if not word in stopWords]
    return (len(tokenizedText)-len(filteredArray))/len(tokenizedText)

def main():
    rawTexts = getTexts()
    filenames = getFilenames()

    tokenizedTexts = list(map(tokenize, rawTexts))
    unpunctuatedTexts = list(map(unpunctuate, tokenizedTexts))
    textLengths = list(map(len, unpunctuatedTexts))

    tokenFreqDists = list(map(getTokenFreqDist, unpunctuatedTexts))
    sortedFreqDists = list(map(sortFreqDist, tokenFreqDists))

    stopWordFreqs = list(map(getStopWordFrequency, tokenizedTexts))
    
    #print(wordFrequency(unpunctuatedTexts[0]))

    #sortedFreqDists[0].plot(15,title="Frequency Distribution of text 1")

    #print(tokenFreqDists[0].values())
    #print(sortedFreqDists[0].values(), sortedFreqDists[1].values())

    array1 = array(list(sortedFreqDists[0].values()))
    array2 = array(list(sortedFreqDists[1].values()))

    kolmogorovResult = stats.ks_2samp(array1, array2).pvalue
    
    #  Kolmogorov-Smirnov
    # https://towardsdatascience.com/how-to-compare-two-distributions-in-practice-8c676904a285
    
    ##    The k-s test returns a D statistic and a p-value corresponding to the D statistic.
    ##    The D statistic is the absolute max distance (supremum) between the CDFs of the two samples.
    ##    The closer this number is to 0 the more likely it is that the two samples were drawn from the same distribution.
    ##    The p-value returned by the k-s test has the same interpretation as other p-values.
    ##    You reject the null hypothesis that the two samples were drawn from the same distribution if the p-value is less than your significance level.

    results = [[filenames[i], textLengths[i], kolmogorovResult, stopWordFreqs[i]] for i in range(len(rawTexts))]


    ### Temporairement
    ### Moyenne sans poids des résultats entre 0 et 1
    #### nombre de mots
    #### kolmogorov
    #### stop word freqs

    averagedResults = [min(textLengths)/max(textLengths),       ## total words
               kolmogorovResult,                                  ## kolmogorov
               min(stopWordFreqs)/max(stopWordFreqs)    ## stop word frequencies
               ]

    results.append(["Similarité"] + averagedResults)

    print(tabulate(results, headers=["Corpus","# de mots", "Kolmogorov distribs de fréq de token", "Fréquence de stop words"], tablefmt="fancy_grid"))
    finalProbability = statistics.mean(averagedResults) * 100

    print("\n\nProbability of same author : %.2f%%\n" % (finalProbability))
if __name__== "__main__":
  main()
