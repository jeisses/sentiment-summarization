from mg_lda import MgLda
import data
import time
import cPickle as pickle
import numpy as np
import make_model
import os

updates = []
log_lik = []


def sentence_probability(model):
    """ Get probabilities for all sentences in the data """

    all_p = []
    for d,doc in enumerate(model.docs):
        p_sent = []
        for s,sent in enumerate(model.sentences):
            words = np.where(np.asarray(l.sent_idx[d]) == s)[0]
            if (len(words) == 0):
                continue


            p_v_r_z = np.zeros(len(model.label_v_r_z))
            for i in words:
                p_v_r_z += model.p_v_r_z(d, i, model.docs[d][i])
            p_sent.append(p_v_r_z / len(words))
        all_p.append(p_sent)

    return all_p

def find_best_topic(topic, pp, model):
    """
    Find most probable sentences for a local topic
    from the sentence probabilities
    (A bit hardcoded, still testing. Returns the most probable sentence
    but the others are not necesarily the next mos probable.. WIP)
    """

    labels = model.label_v_r_z
    sentences = []

    for d,p_doc in enumerate(pp):
        for s,p_sent in enumerate(p_doc):
            if len(model.sentences[d][s]) < 3:
                continue

            best_idx = np.argmax(p_sent)
            lab = labels[best_idx]
            prob = p_sent[best_idx]

            if lab[1] == 1 and lab[2] == topic:
                sentences += [(prob, s, d)]

    sentences = sorted(sentences)

    return sentences[-10:]


def extract_sentence(model, d, s):
    dir = "./data/all/"
    filenames = [filename for filename in os.listdir(dir) if filename.endswith(".txt")]
    
    raw = open(dir + filenames[d]).read()

    sentence = model.sentences[d][s]

    first = sentence[0]
    firstIdx = 0

    for i in range(0, len(raw) - len(first)):
        if raw[i:i + len(first)] == first:
            firstIdx = i
            break


    last = sentence[-1]

    lastIdx = len(raw)

    for i in range(len(raw), len(last), -1):
        if raw[i - len(last):i] == last:
            lastIdx = i
            break


    return raw[firstIdx:lastIdx]

    


# Get most probable sentences for each topic
l = make_model.l
print "Calculating sentence probabilities..."
sent_prob = sentence_probability(l)
print "Done! Printing topics..."
for z in range(l.K_loc):
    sentences = find_best_topic(z, sent_prob, l)

    for prob, s, d in sentences:
        print "Topic %d: doc %d sent %s "%(z, d, l.sentences[d][s])

        # Helps with searching:
        # print extract_sentence(l, d, s)
