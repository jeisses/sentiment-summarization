import os
import nltk
import string
import numpy as np


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
        updated = 0
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
                new_z = np.random.multinomial(1, p_z/sum(p_z)).argmax()

                if new_z != cur_z:
                    updated += 1

                self.assignment[i][w] = new_z
                self.nwz[cur_w, new_z] += 1
                self.ndz[i, new_z] += 1
                self.nz[new_z] += 1
        print "Updated %d"%updated


def parse_dir(dir="./"):
    """
    Create vocabulary from directory. Parses all .txt files.
    Returns documents as tokens, lookup table of each word
    in the vocabulary and the vocabulary.
    """
    try:
        nltk.corpus.stopwords.abspath('english')
    except LookupError:
        nltk.download('stopwords')

    docs = list() # list of the words for each review
    vocab = set() # the vocabulary
    stopwords = nltk.corpus.stopwords.words('english') \ 
                + list(string.punctuation) # these words are ignored
    tokenizer = nltk.wordpunct_tokenize # callable to tokenize
    files = [f for f in os.listdir(dir) if f.endswith(".txt")]
    for f in files:
        raw = open(dir + f).read()
        tokens = tokenizer(raw)
        words = [w.lower() for w in tokens if w not in stopwords]
        docs.append(words)
        vocab.update(words)
    vocab = list(vocab)

    word_idx = {}
    for i,w in enumerate(vocab):
        word_idx[w] = i
 
    return docs, word_idx,vocab


# EXAMPLE
print "Parsing dir.."
docs, word_idx, vocab = parse_dir("./data/")
print "Done"
print " ====== "
print "Setting up LDA.."
l = Lda(10, docs, vocab, word_idx)
print "Done. Top words for each topic:"
l.top_words(10)
print "Training for 500 iterations..."
for i in range(0, 500):
    l.update()
    if i % 10 == 0:
        print " -- iteration %d ---"%i
print "Done. Top words for reach topic:"
l.top_words(10)
