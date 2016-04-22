__author__ = 'Dimitri Zhang'
import numpy as np
import os.path
import os
import nltk
import string
import itertools
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from time import time

def loadData(path):
    stack = [path]
    count = 0
    corpus = []
    while count < 10000:
        top = stack[-1]
        stack.pop()
        if not os.path.isdir(top) and os.path.basename(top) != '.DS_Store':
            count += 1
            if count % 10000 == 0:
                print(count)
                print(top)
            # print(count)
            # corpus.append(' '.join(extractKW(top)))
            corpus.append(' '.join(preprocess(top)))
        else:
            subdirs = os.listdir(top)
            for d in subdirs:
                if d != '.DS_Store':
                    stack.append(top + '/' + d)
    return corpus

def preprocess(path):
    with open(path, 'rb') as f:
        b = f.read()
    s = b.decode('utf8', errors='ignore')
    subjectIndex = s.find('Subject:')
    mimeVersIndex = s.find('Mime-Version')
    subject = s[subjectIndex + 8: mimeVersIndex].strip()
    contentIndex = s.find('X-FileName:')
    while s[contentIndex] != '\n':
        contentIndex += 1
    content = s[contentIndex + 1:].strip()
    content = content + subject
    good_tags = set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])
    stopWords = set(nltk.corpus.stopwords.words('english'))
    punct = set(string.punctuation)
    sentences = nltk.sent_tokenize(content)
    tagged_words = itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent)
                                                                    for sent in sentences))
    porterStemmer = nltk.PorterStemmer()
    candidates = [word.lower() for word, tag in tagged_words if tag in good_tags and word.lower not in stopWords 
        and all(char not in punct for char in word) and len(word) > 3]  
    return candidates


def extractKW(path):
    punct = set(string.punctuation)
    stopWords = set(nltk.corpus.stopwords.words('english'))
    good_tags = set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])
    # grammar = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
    with open(path, 'rb') as f:
        b = f.read()
    s = b.decode('utf8', errors='ignore')
    subjectIndex = s.find('Subject:')
    mimeVersIndex = s.find('Mime-Version')
    subject = s[subjectIndex + 8: mimeVersIndex].strip()
    contentIndex = s.find('X-FileName:')
    while s[contentIndex] != '\n':
        contentIndex += 1
    content = s[contentIndex + 1:].strip()
    content = content + subject
    sentences = nltk.sent_tokenize(content)
    # print(sentences)
    porterStemmer = nltk.PorterStemmer()
    candidates = list(itertools.chain.from_iterable(nltk.word_tokenize(sentence) for sentence in sentences))
    words = [word.lower() for word in candidates            
        if word not in stopWords and all(char not in punct for char in word)]
    # print(words)
    return words

def showTopics(component, featureNames, numTopWords):
    for topicIndex, wordDistribution in enumerate(component):
        print(type(wordDistribution))            
        index = np.argsort(wordDistribution)
        print("Topic #%d:" % topicIndex)
        print(','.join(featureNames[index[-numTopWords:]]))

def sortByValue(featMap):
    sortedFeatMap = sorted(featMap.items(), key = lambda x:x[1], reverse = True)
    return sortedFeatMap    

def main():
    # corpus = loadData('/Users/zhangtianyao/Documents/maildir')
    # with open('5000mails.txt', 'w') as f:
    #     for line in corpus:
    #         f.write(line + '\n')
    t0 = time()
    corpus = []
    with open('5000mails.txt', 'r') as f:
        count = 0
        for line in f:
            corpus.append(line)
            count += 1
            if count == 5000:
                break
    print("data loaded in %0.3fs." % (time() - t0))
    t0 = time()
    vectorizer = CountVectorizer(max_df = 0.95, min_df = 2)
    X = vectorizer.fit_transform(corpus)
    print(X.shape)
    # print(type(vectorizer.get_feature_names()))
    l = np.array(vectorizer.get_feature_names())
    print(type(l))
    print(l[:10])
    print("transformed into feature matrix in %0.3fs." % (time() - t0))
    t0 = time()
    lda = LatentDirichletAllocation(n_topics = 10, random_state = 0)
    lda.fit(X)
    print("lda model trained in %0.3fs." % (time() - t0))
    lda.components_
    showTopics(lda.components_, l, 10)
    # transformer = TfidfTransformer()
    # transformedX = transformer.fit_transform(X)
    # print(transformedX.shape)
    # km = KMeans(n_clusters = 10)
    # model = km.fit(transformedX)
    # label = model.predict(transformedX)
    # print(label)

if __name__ == '__main__':
    main()
