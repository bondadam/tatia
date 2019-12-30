import nltk
import os
import statistics
import re
#import matplotlib
from scipy import stats ##requires numpy + mkl
from scipy import array
from collections import Counter
from nltk.corpus import stopwords
from tabulate import tabulate
import csv
import spacy

##NLTK packages to install : punkt

def getTexts():
    """
    Reads text files present in current directory, stores them in strings,
    closes the files then returns strings. Fails if >2 or <2 text files.
    """
    allTexts = []
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    textFiles = []
    for f in files:
        if f[-4:] == ".txt":
            textFiles.append(f)
    for file in textFiles:
        with open(file, 'r',encoding="utf-8") as f1:
            allTexts.append(f1.read())
    return allTexts


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

def getTokenizedSentences(textString):
    """Input: text string
    returns array of sentences, each sentence an array of tokenized words"""
    sentences = re.split('\.+|\?+|\!+|[\?\!]+| [\r\n]+',textString)
    return [nltk.word_tokenize(sentence, language='french') for sentence in sentences]


def getTokenFreqDist(tokenizedText):
    """Input: tokenized text
    returns array of token length frequency distribution"""
    tokenLengths = [len(token) for token in tokenizedText]
    return nltk.FreqDist(tokenLengths)

def getSentenceLengthFreqDist(tokenizedSentences):
    """Input: tokenized sentences
    returns array of sentence length frequency distribution"""
    sentenceLengths = [len(sentence) for sentence in tokenizedSentences]
    return nltk.FreqDist(sentenceLengths)

def getSentenceMeanLength(tokenizedSentences):
    """Input: tokenized sentences
    returns average length (nb words)"""
    sentenceLengths = [len(sentence) for sentence in tokenizedSentences]
    return statistics.mean(sentenceLengths)

def sortFreqDist(freqDist):
    return dict(sorted(freqDist.items()))

def compareTokenFreqDists(freqDist1, freqDist2):
    pass

def getStopWordFrequency(tokenizedText):
    stopWords = set(stopwords.words('french'))
    filteredArray = [word for word in tokenizedText if not word in stopWords]
    return (len(tokenizedText)-len(filteredArray))/len(tokenizedText)

def getLongWordsSimilarity(unpunctuatedTexts):
    """Input: unpunctuated tokenized texts
    keep only words > 6 characters
    return distance between sets
    """
    filteredTexts = []
    for text in unpunctuatedTexts:
        filteredText= set([word for word in text if len(word) >= 5])
        filteredTexts.append(filteredText)
    return nltk.jaccard_distance(filteredTexts[0], filteredTexts[1])
    

def getVerbRatio(rawText):
    nlp = spacy.load("fr_core_news_sm")
    doc = nlp(rawText)
    nbVerbs = sum(value.pos_ == 'VERB' for value in doc)
    return nbVerbs/len(doc)

def analyze(rawTexts):
    filenames = getFilenames()

    #print([(w.text, w.pos_) for w in doc])


    tokenizedTexts = list(map(tokenize, rawTexts))
    unpunctuatedTexts = list(map(unpunctuate, tokenizedTexts))
    textLengths = list(map(len, unpunctuatedTexts))


    tokenFreqDists = list(map(getTokenFreqDist, unpunctuatedTexts))
    sortedTokenFreqDists = list(map(sortFreqDist, tokenFreqDists))

    stopWordFreqs = list(map(getStopWordFrequency, tokenizedTexts))

    tokenizedSentences = list(map(getTokenizedSentences, rawTexts))
    sentenceFreqDists = list(map(getSentenceLengthFreqDist, tokenizedSentences))
    sortedSentenceFreqDists = list(map(sortFreqDist, sentenceFreqDists))

    meanSentenceLengths = list(map(getSentenceMeanLength, tokenizedSentences))
    verbRatios = list(map(getVerbRatio, rawTexts))

    longWordsSimilarity = getLongWordsSimilarity(unpunctuatedTexts)
    #print(wordFrequency(unpunctuatedTexts[0]))

    #print(tokenizedSentences[1])

    #sortedFreqDists[0].plot(15,title="Frequency Distribution of text 1")

    array1 = array(list(sortedTokenFreqDists[0].values()))
    array2 = array(list(sortedTokenFreqDists[1].values()))
    

    tokenFreqDistResult = stats.ks_2samp(array1, array2).pvalue

    array1 = array(list(sortedSentenceFreqDists[0].values()))
    array2 = array(list(sortedSentenceFreqDists[1].values()))

    sentenceFreqDistResult = stats.ks_2samp(array1, array2).pvalue


    #  Kolmogorov-Smirnov
    # https://towardsdatascience.com/how-to-compare-two-distributions-in-practice-8c676904a285
    
    ##    The k-s test returns a D statistic and a p-value corresponding to the D statistic.
    ##    The D statistic is the absolute max distance (supremum) between the CDFs of the two samples.
    ##    The closer this number is to 0 the more likely it is that the two samples were drawn from the same distribution.
    ##    The p-value returned by the k-s test has the same interpretation as other p-values.
    ##    You reject the null hypothesis that the two samples were drawn from the same distribution if the p-value is less than your significance level.

    ### Temporairement
    ### Moyenne sans poids des résultats entre 0 et 1
    #### nombre de mots
    #### kolmogorov
    #### stop word freqs

    baseResults = [[tokenFreqDistResult,tokenFreqDistResult],
                   [stopWordFreqs[0], stopWordFreqs[1]],
                   [sentenceFreqDistResult, sentenceFreqDistResult],
                   [meanSentenceLengths[0],meanSentenceLengths[1]],
                   [longWordsSimilarity, longWordsSimilarity],
                   [verbRatios[0], verbRatios[1]]]

    averagedResults = [tokenFreqDistResult,                                  ## kolmogorov
               min(stopWordFreqs)/max(stopWordFreqs),    ## stop word frequencies
               sentenceFreqDistResult, ## also kolmogorov
               min(meanSentenceLengths)/max(meanSentenceLengths),  ## sentence lengths in words
               1 - longWordsSimilarity,
               min(verbRatios)/max(verbRatios)] ## %verb text 1 / %vebr text2

    headerTitles = ["K-S distribs de fréq de token", "Fréquence de stop words", "K-S distribs de fréq de phrase", "Longueur moyenne de phrase", "Distance Jaccard mots > 7 lettres", "% de verbes"]
    results = [[headerTitles[i], baseResults[i][0], baseResults[i][1], averagedResults[i]] for i in range(len(baseResults))]


    #print(tabulate(results, headers=["", filenames[0], filenames[1], "Similarité"], tablefmt="fancy_grid"))
    #finalProbability = statistics.mean(averagedResults) * 100

    #print("\n\nProbability of same author : %.2f%%\n" % (finalProbability))
    #return finalProbability
    return averagedResults

def main():
    allTexts = getTexts()
    with open('data.csv', 'w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        #headerTitles = ["titre 1", "titre 2", "K-S distribs de fréq de token", "Fréquence de stop words", "K-S distribs de fréq de phrase", "Longueur moyenne de phrase", "Distance Jaccard mots > 5 lettres", "% de verbes", "Meme auteur"]
        headerTitles = ["K-S distribs de fréq de token", "Fréquence de stop words", "K-S distribs de fréq de phrase", "Longueur moyenne de phrase", "Distance Jaccard mots > 5 lettres", "% de verbes", "Meme auteur"]
        writer.writerow(headerTitles)
        for i in range(len(allTexts)):
            for j in range(i+1, len(allTexts)):
                ## Couples de 2 textes
                text1 = allTexts[i]
                text2 = allTexts[j]

                splitText1 = text1.split("\n")
                splitText2 = text2.split("\n")
                
                author1 = splitText1[0]
                author2 = splitText2[0]

                title1 = splitText1[1]
                title2 = splitText2[1]

                rawText1 = "\n".join(splitText1[3:])
                rawText2 = "\n".join(splitText2[3:])
                
                print(author1 + " - " + title1 + " VS " + author2 + " - " + title2 + " : ") #+ str(analyze([rawText1, rawText2])))
                result = analyze([rawText1, rawText2])
                sameAuthor = author1.lower() == author2.lower()
                writer.writerow(result + [int(sameAuthor)])
                file.flush()
                ### flush pour enregistrer les changements
if __name__== "__main__":
  main()
  input("Press Enter to continue...")
