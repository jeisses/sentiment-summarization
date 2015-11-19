from __future__ import division

import nltk
import random
from math import log
import os
import heapq
import string
import lda
import numpy


def create_vocabulary(X):
    print "Creating vocabulary."
    vocabulary = []
    visited = set()
    for x in X:
        for word in x:
            if word not in visited:
                vocabulary.append(word)
                visited.add(word)

    return vocabulary

def create_count_matrix(X, vocabulary):
    print "Creating count matrix."
    mapping = {}
    index = 0
    for word in vocabulary:
        mapping[word] = index
        index += 1

    newX = numpy.zeros((len(X), len(vocabulary)), dtype=int)

    for i, x in enumerate(X):
        for word in x:
            newX[i][mapping[word]] += 1

    return newX




def shuffle(l):
    random.shuffle(l)
    return l

class LDA(object):
    def __init__(self, X, K, alpha = 0.5, beta = 0.5):
        self.X = X
        self.V = len(X[0])
        self.K = K
        self.Z = numpy.random.randint(K, size=(len(X), len(X[0])))

        self.count_topic_word = numpy.zeros((K, self.V), dtype=int)
        self.count_topic = numpy.zeros(K, dtype=int)
        self.count_document_topic = numpy.zeros((len(X), K), dtype=int)
        self.count_document = numpy.zeros(len(X), dtype=int)

        for i in range(len(X)):
            for j in range(len(X[i])):
                k = self.Z[i][j]
                word_count = X[i][j]
                self.count_topic_word[k][j] += word_count

                self.count_topic[k] += word_count
                self.count_document_topic[i][k] += word_count
                self.count_document[i] += word_count

        self.alpha = alpha
        self.beta = beta

        self.train()

    def train(self):
        X = self.X
        Z = self.Z
        K = self.K

        for iteration in range(15):
            print "Iteration: ", iteration

            assignments_changed = 0

            for i in shuffle(range(len(X))):
                document = X[i]

                for word in shuffle(range(len(document))):
                    word_count = document[word]

                    topic = Z[i][word]

                    self.count_topic_word[topic][word] -= word_count
                    self.count_topic[topic] -= word_count
                    self.count_document_topic[i][topic] -= word_count
                    self.count_document[i] -= word_count


                    highest_p = -4096.
                    highest_k = 0
                    for k in range(K):
                        p = self.p_topic_word(i, word, k)
                
                        if p > highest_p:
                            highest_p = p
                            highest_k = k

                    if highest_k != topic:
                        assignments_changed += 1

                    Z[i][word] = highest_k

                    self.count_topic_word[highest_k][word] += word_count
                    self.count_topic[highest_k] += word_count
                    self.count_document_topic[i][highest_k] += word_count
                    self.count_document[i] += word_count

            print "Assignments changed:", assignments_changed
            if assignments_changed == 0:
                break

    def p_topic_word(self, i, word, topic):
        alpha = self.alpha
        beta = self.beta
        V = self.V
        K = self.K

        p = log(beta + self.count_topic_word[topic][word]) - log(beta * V + self.count_topic[topic]) + log(alpha + self.count_document_topic[i][topic]) - log(alpha * K + self.count_document[i])

        return p

    def word_per_topic(self):
        result = []
        for k in range(self.K):
            l = []
            for word in range(self.V):
                l.append((self.count_topic_word[k][word], word))

            result.append([item[1] for item in heapq.nlargest(10, l)])
        return result

if __name__ == '__main__':
    path = 'review_polarity/txt_sentoken/pos'
    X = []
    remove = nltk.corpus.stopwords.words('english') + list(string.punctuation) + ['``', "\'s"]

    t = 0
    for filename in os.listdir(os.getcwd() + '/' + path):
        if t > 20:
            break

        f = open(path + "/" + filename)
        X.append([word for word in nltk.word_tokenize(f.read()) if word not in remove])

        t += 1
        

    vocabulary = create_vocabulary(X)
    X = create_count_matrix(X, vocabulary)

    # model = lda.LDA(n_topics=20, n_iter=500, random_state=1)
    # model.fit(X)  # model.fit_transform(X) is also available
    # topic_word = model.topic_word_  # model.components_ also works
    # n_top_words = 8

    # for i, topic_dist in enumerate(topic_word):
    #     topic_words = numpy.array(vocabulary)[numpy.argsort(topic_dist)][:-(n_top_words+1):-1]
    #     print('Topic {}: {}'.format(i, ' '.join(topic_words)))

    lda = LDA(X, 10)



    result = lda.word_per_topic()

    for i in range(len(result)):
        print "Topic {}: {}".format(i, [vocabulary[index] for index in result[i]])

