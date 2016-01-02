from __future__ import division
import string
import numpy as np
import data

from math import log

import matplotlib.pyplot as plt

class Lda:
    alpha = 0.5
    beta = 0.5

    def __init__(self, n_topics, docs, vocab, word_idx):
        """ Create counts and random initial assignment """
        n_docs = len(docs)
        vocab_size = len(vocab)

        self.docs = docs
        self.vocab = vocab
        self.word_idx = word_idx # lookup table of word indicis in vocab
        self.n_topics = n_topics
        self.ndz = np.zeros((n_docs, n_topics)) # times d has been assigned to each topic
        self.nwz = np.zeros((vocab_size, n_topics)) # times w has been assigned to each topic
        self.nz = np.zeros(self.n_topics) # times z has been assigned to any word

        self.random_assignment()

        self.loglikelihood = []

    def random_assignment(self):
        """ Random assign topics """
        self.assignment = []
        for i,doc in enumerate(self.docs):
            self.assignment.append([])
            for w,word in enumerate(doc):
                z = np.random.randint(0, self.n_topics)
                cur_w = word_idx[word]
                self.ndz[i,z] += 1
                self.nwz[cur_w,z] += 1
                self.nz[z] += 1
                self.assignment[i].append(z)

    def top_words(self, n):
        """ Print the top x words for each topic """
        for k in xrange(self.n_topics):
            idcs = np.argpartition(self.nwz[:,k],-n)[-n:]
            words = []
            for w in idcs:
                words.append(self.vocab[w])
            print "Topic %d: %s"%(k, words)

    def update(self):
        """ Perform 1 step of Gibbs sampling """
        loglikelihood = 0.0

        for i,doc in enumerate(self.docs):
            for w,word in enumerate(doc):
                # remove this word from the counts
                cur_w = self.word_idx[word]
                cur_z = self.assignment[i][w]
                self.nwz[cur_w, cur_z] -= 1
                self.ndz[i, cur_z] -= 1
                self.nz[cur_z] -= 1
 
                # calculate p(k=z_ij)
                p_zw = (self.beta + self.nwz[cur_w,:]) / (len(self.vocab)*self.beta + self.nz)
                p_zd = (self.alpha + self.ndz[i]) / (self.n_topics*self.alpha + len(doc))
                p_z = p_zw * p_zd

                p_z /= sum(p_z)

                new_z = np.random.multinomial(1, p_z).argmax()

                loglikelihood += log(p_z[new_z])

                self.assignment[i][w] = new_z
                self.nwz[cur_w, new_z] += 1
                self.ndz[i, new_z] += 1
                self.nz[new_z] += 1

        print "Loglikelihood %f" % loglikelihood
        self.loglikelihood.append(loglikelihood)
        




# EXAMPLE
print "Parsing dir.."
docs, vocab, word_idx, _, _ = data.parse_dir("./data/all/")
# docs, vocab, word_idx, _, _ = data.parse_dir("/Users/jeisses/Documents/datasets/nlp/movie/review_polarity/txt_sentoken/all/")
print "Done"
print " ====== "
print "Setting up LDA.."
l = Lda(10, docs, vocab, word_idx)

print "Training for 200 iterations..."
for i in range(0, 200):
    l.update()
    if i % 10 == 0:
        print " -- iteration %d ---"%i
print "Done. Top words for reach topic:"
l.top_words(10)

loglikelihood = [item / 100000 for item in l.loglikelihood]

plt.plot(loglikelihood, 'r-')
plt.xlabel('Iterations')
plt.ylabel('Loglikelihood x 10^5')
plt.show()

