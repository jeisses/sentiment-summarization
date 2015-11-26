from __future__ import division

import nltk
import random
import os
import heapq
import string
import numpy
import DataModule

def shuffle(l):
    return l
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

        for iteration in range(200):
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
                    
                    p = self.p_word(i, word)
                    new_topic = numpy.argmax(numpy.random.multinomial(1, p))

                    if topic != new_topic:
                        assignments_changed += 1


                    Z[i][word] = new_topic

                    self.count_topic_word[new_topic][word] += word_count
                    self.count_topic[new_topic] += word_count
                    self.count_document_topic[i][new_topic] += word_count
                    self.count_document[i] += word_count

            print "Assignments changed:", assignments_changed

    def p_word(self, i, word):
        alpha = self.alpha
        beta = self.beta
        V = self.V
        K = self.K

        p = (beta + self.count_topic_word[:,word]) / (beta * V + self.count_topic) * (alpha + self.count_document_topic[i]) / (alpha * K + self.count_document[i])
        

        return p / numpy.sum(p)


    def word_per_topic(self):
        result = []
        for k in range(self.K):
            l = []
            for word in range(self.V):
                l.append((self.count_topic_word[k][word], word))

            result.append([item[1] for item in heapq.nlargest(10, l)])
        return result

if __name__ == '__main__':
    dataHandler = DataModule.DataHandler()

    X = dataHandler.get_count_matrix()

    lda = LDA(X, 5)

    print [[dataHandler.get_word_with_id(w) for w in topic_words] for topic_words in lda.word_per_topic()]



