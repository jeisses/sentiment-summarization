from mg_lda import MgLda
import data
import time
import cPickle as pickle
import numpy as np
import make_model

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
    best_prob = -9999.0
    best_sent = [0]*5 # grab 5 sentences for now
    best_doc  = [0]*5 
    for d,p_doc in enumerate(pp):
        for s,p_sent in enumerate(p_doc):
            best_idx = np.argmax(p_sent)
            lab = labels[best_idx]
            prob = p_sent[best_idx]
            if lab[1] == 1 and lab[2] == topic:
                if prob >= best_prob:
                    best_sent[4] = best_sent[3] 
                    best_sent[3] = best_sent[2] 
                    best_sent[2] = best_sent[1] 
                    best_sent[1] = best_sent[0] 
                    best_sent[0] = s
                    best_doc[4] = best_doc[3] 
                    best_doc[3] = best_doc[2] 
                    best_doc[2] = best_doc[1] 
                    best_doc[1] = best_doc[0] 
                    best_doc[0] = d
                    best_prob = prob
    return best_prob, best_doc, best_sent


# Get most probable sentences for each topic
l = make_model.l
print "Calculating sentence probabilities..."
sent_prob = sentence_probability(l)
print "Done! Printing topics..."
for z in range(l.K_loc):
    _,doc,sent= find_best_topic(z, sent_prob, l)
    for i,d in enumerate(doc):
        print "Topic %d sent %s "%(z, l.sentences[d][sent[i]])
