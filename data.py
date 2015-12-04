import nltk
import os
import string
import numpy as np
from nltk.tokenize import sent_tokenize

MIN_SENTENCE_LEN = 2

def check_nltk_data():
    """ Download needed data from NLTK """
    try:
        nltk.corpus.stopwords.abspath('english')
        nltk.data.load('tokenizers/punkt/english.pickle')
    except LookupError:
        nltk.download('stopwords')
    except ValueError:
        nltk.download('punkt')


def get_stopwords():
    """ Stopwords are ignored """
    stopwords = nltk.corpus.stopwords.words('english') \
                + list(string.punctuation)
    return stopwords

def parse_document(content, stopwords):
    """ Parse a document sentence by sentence """
    sentences = list()
    word_sentence_map = []
    words = list()
    line = 0
    raw_sentences = sent_tokenize(content)
    for s in raw_sentences:
        # Skip small sentences
        if len(s) < MIN_SENTENCE_LEN:
            continue
        else:
            line += 1

        tokens = nltk.wordpunct_tokenize(s)
        sent_words = [w.lower() for w in tokens if w not in stopwords]
        sentences.append(sent_words)
        word_sentence_map.extend([line-1]*len(sent_words))
        words.extend(sent_words)

    return words, word_sentence_map, sentences


def parse_dir(dir="./"):
    """
    Create vocabulary from directory. Parses all .txt files.

    Returns 5 values:
     - documents as tokens
     - the vocabulary
     - lookup table of each word in the vocabulary
     - sentence index for each word in the document
    """
    check_nltk_data()
    stopwords = get_stopwords()

    docs = list() # list of the words for each review
    word_sentence_map = list() # word mapping to sentences for each document
    sentences = list()
    vocab = set() # the vocabulary
    files = [f for f in os.listdir(dir) if f.endswith(".txt")]
    for f in files:
        raw = open(dir + f).read()
        words, sentence_map, doc_sent = parse_document(raw, stopwords)
        docs.append(words)
        sentences.append(doc_sent)
        word_sentence_map.append(sentence_map)
        vocab.update(words)
    vocab = list(vocab)

    # Create a mapping from words to vocabulary indicis 
    word_idx = {}
    for i,w in enumerate(vocab):
        word_idx[w] = i
 
    return docs, vocab, word_idx, word_sentence_map, sentences


def analyze_data(docs, vocab, word_idx, word_sentence_map, sentences):
    sents_per_doc = np.array([len(s) for s in sentences])
    words_per_doc = np.array([len(d) for d in docs])

    print "Reviews {:,d}".format(len(docs))
    print "Sentences {:,d}".format(np.sum(sents_per_doc))
    print "Words {:,d}".format(np.sum(words_per_doc))
    print "Sentences per review %.1f"%np.mean(sents_per_doc)
    print "Words per review %.1f"%np.mean(words_per_doc)

