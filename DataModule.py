from __future__ import division

import nltk
import string
import os
import numpy


class DataHandler(object):
    def __init__(self):
        self.X, self.X_sentenced = self._readData()
        self.vocabulary = self._create_vocabulary(self.X)
        self.mapping_word_to_number, self.mapping_number_to_word = self._create_mapping(self.vocabulary)

    def get_word_for_id(self, identifier):
        return self.mapping_number_to_word[identifier]

    def get_id_for_word(self, word):
        return self.mapping_word_to_number[word]
    
    def get_data_count_matrix(self):
        count_matrix = numpy.zeros((len(self.X), len(self.vocabulary)), dtype=int)

        for i in range(len(self.X)):
            x = self.X[i]
            for j in range(len(x)):
                word = x[j]
                count_matrix[i][self.mapping_word_to_number[word]] += 1

        return count_matrix

    def get_data_sentenced(self):
        return self.X_sentenced


    def _readData(self):
        print "Reading data from files"
        path = 'data'
        X_sentenced = []
        X = []

        remove = set(nltk.corpus.stopwords.words('english') + list(string.punctuation) + ['``', "\'s", '--'])

        for filename in os.listdir(os.getcwd() + '/' + path):
            f = open(path + "/" + filename)
            x = []
            x_sentenced = []
            sentence = []
            for word in nltk.word_tokenize(f.read()):
                if word not in remove:
                    x.append(word)

                if word == '.' or word == '?' or word == '!' or word == ':':
                    if sentence:
                        x_sentenced.append(sentence)
                        sentence = []
                else:
                    if word not in remove:
                        sentence.append(word)
            X.append(x)
            X_sentenced.append(x_sentenced)

        return X, X_sentenced

    def _create_vocabulary(self, X):
        print "Creating vocabulary."
        vocabulary = []
        visited = set()
        for x in X:
            for word in x:
                if word not in visited:
                    vocabulary.append(word)
                    visited.add(word)

        return vocabulary

    def _create_mapping(self, vocabulary):
        mapping_word_to_number = {}
        mapping_number_to_word = [0 for i in range(len(vocabulary))]

        for t, word in enumerate(list(vocabulary)):
            mapping_number_to_word[t] = word
            mapping_word_to_number[word] = t

        return mapping_word_to_number, mapping_number_to_word

    def _convert_words_to_ids(self, X, mapping):
        newX = []
        for x in X:
            newx = []
            for sentence in x:
                newSentence = []
                for word in sentence:
                    newSentence.append(mapping[word])
                newx.append(newSentence)
            newX.append(newx)
        return newX


if __name__ == '__main__':
    dataHandler = DataHandler()


