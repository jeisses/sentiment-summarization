import string
import numpy as np
import data
import time

class MgLda:
    updates = 0    # Updated topics in the last iteration
    log_lik = 0.0  # Log likelihood of the current model

    def __init__(self, n_local_topics, n_global_topics, docs, vocab, word_idx, sentences, sent_idx):
        """ Create counts and random initial assignment """
        self.alpha_loc     = 0.1
        self.alpha_gl      = 0.1
        self.alpha_mix_loc = 0.1 # why 2 alpha_mix ??
        self.alpha_mix_gl  = 0.1
        self.beta_loc      = 0.1
        self.beta_gl       = 0.1
        self.gamma         = 0.1
        self.T             = 3 # sliding window size

        print "INIT"

        n_docs     = len(docs)
        vocab_size = len(vocab)

        self.docs     = docs
        self.vocab    = vocab
        self.sentences = sentences
        self.sent_idx = sent_idx
        self.word_idx = word_idx # lookup table of word indicis in vocab
        self.K_loc    = n_local_topics
        self.K_gl     = n_global_topics

        self.v_d_s_n  = []  # window assignment for each word in a document
        self.z_d_s_n  = []  # topic assignment for each word in a document
        self.r_d_s_n  = []  # glob (0) / loc (1) assignment for each word in a document

        self.n_d_s    = []  # nr of words in sentence s
        self.n_d_s_v  = []  # nr of words in sentence s assigned to window v
        self.n_d_v    = []  # nr of words in window v

        self.n_d_v_gl  = []  # nr of words in window v assigned to global topics
        self.n_d_v_loc = []  # nr of words in window v assigned to local topics

        self.n_d_gl_z   = np.zeros((n_docs, self.K_gl))  # times d has been assigned to global topic
        self.n_d_gl     = np.zeros(n_docs)          # times d has been assigned

        self.n_d_v_loc_z = []  # times window v has been assigned local topic

        self.n_gl_z_w  = np.zeros((self.K_gl, vocab_size))  # times w has been assigned to global topic
        self.n_loc_z_w = np.zeros((self.K_loc, vocab_size)) # times w has been assigned to local topic

        self.n_gl_z  = np.zeros(self.K_gl)  # times global z has been assigned to any word
        self.n_loc_z = np.zeros(self.K_loc) # times local z has been assigned to any word

        # Cache the label lookup table
        # Used in the update hook to quickly find labels and responsibilities
        # for a sample.
        self.label_v_r_z = []
        for v_t in range(self.T):
            for z_t in range(self.K_gl):
                label = [v_t, 0, z_t]
                self.label_v_r_z.append(label)
        for v_t in range(self.T):
            for z_t in range(self.K_loc):
                label = [v_t, 1, z_t]
                self.label_v_r_z.append(label)


        self.random_assignment()

    def random_assignment(self):
        """ Random assign topics """
        self.assignment = []

        print "DOING random assignment"

        for i,doc in enumerate(self.docs):
            v_d = [] # window id for each word relative to its sentence
            r_d = [] # glob / loc for each word
            z_d = [] # topic id for each word

            self.n_d_s.append([len(s) for s in self.sentences[i]])
            self.n_d_s_v.append([])

            n_windows = self.T+len(self.sentences[i])

            for s in self.sentences[i]:
                self.n_d_s_v[i].append(np.asarray([0 for a in range(0, self.T)]))
            self.n_d_v.append(np.asarray([0 for a in range(0, n_windows)]))

            self.n_d_v_gl.append(np.asarray([0]*n_windows))
            self.n_d_v_loc.append(np.asarray([0]*n_windows))
            self.n_d_v_loc_z.append(np.zeros((n_windows, self.K_loc)))

            for w,word in enumerate(doc):
                v = np.random.randint(0, self.T) # sliding window for this word, relative to sent
                v_d.append(v)

                r = np.random.randint(0,2) # 0 = global, 1 = local
                r_d.append(r)

                z = 0
                if r == 0:
                    z = np.random.randint(0, self.K_gl)
                else:
                    z = np.random.randint(0, self.K_loc)
                z_d.append(z)

                cur_w = self.word_idx[word]

                s = self.sent_idx[i][w] # sentence nr for this word

                self.n_d_s_v[i][s][v] += 1
                self.n_d_v[i][s+v]    += 1

                if r == 0:
                    self.n_gl_z_w[z][cur_w] += 1
                    self.n_gl_z[z] += 1
                    self.n_d_v_gl[i][s+v] += 1
                    self.n_d_gl_z[i][z] += 1
                    self.n_d_gl[i] += 1
                else:
                    self.n_loc_z_w[z][cur_w] += 1
                    self.n_loc_z[z] += 1
                    self.n_d_v_loc[i][s+v] += 1
                    self.n_d_v_loc_z[i][s+v][z] += 1

            self.v_d_s_n.append(v_d)
            self.r_d_s_n.append(r_d)
            self.z_d_s_n.append(z_d)

    def top_words(self, n, type):
        """
        Print the top x words for each topic. Type is "glob" or "loc"
        """
        if type == "glob":
            K = self.K_gl
        elif type == "loc":
            K = self.K_loc

        for k in xrange(K):
            if type == "glob":
                idcs = np.argpartition(self.n_gl_z_w[k],-n)[-n:]
            elif type == "loc":
                idcs = np.argpartition(self.n_loc_z_w[k],-n)[-n:]

            words = []
            for w in idcs:
                words.append(self.vocab[w])

            if type == "glob":
                print "Global Topic %d: %s"%(k, words)
            elif type == "loc":
                print "Local Topic %d: %s"%(k, words)

    def p_v_r_z(self, d, i, w):
        """
        Calculate the conditional probabilities for all kinds topics for words i.
        Returns array with first global probs for each window, followed by local
        probs for each window.
        """
        W = len(self.vocab)
        s = self.sent_idx[d][i]
        word = self.word_idx[w]

        # probabilities for global topics
        term1 = (self.n_gl_z_w[:,word] + self.beta_gl) / (self.n_gl_z + W*self.beta_gl)
        term2 = (self.n_d_s_v[d][s] + self.gamma) / (self.n_d_s[d][s] + self.T*self.gamma)
        term3 = (self.n_d_v_gl[d][s:s+self.T] + self.alpha_mix_gl) / (self.n_d_v[d][s:s+self.T] + self.alpha_mix_gl + self.alpha_mix_loc)
        term4 = (self.n_d_gl_z[d] + self.alpha_gl) / (self.n_d_gl[d] + self.K_gl*self.alpha_gl)
        score_glob = term1 * np.reshape(term2, (3,1)) * np.reshape(term3, (3,1))* term4
        score_glob = np.asarray(score_glob).reshape(-1)

        # probabilities for local topics
        term1 = (self.n_loc_z_w[:,word] + self.beta_loc) / (self.n_loc_z + W*self.beta_loc)
        term2 = (self.n_d_s_v[d][s] + self.gamma) / (self.n_d_s[d][s] + self.T*self.gamma)
        term3 = (self.n_d_v_loc[d][s:s+self.T] + self.alpha_mix_loc) / (self.n_d_v[d][s:s+self.T] + self.alpha_mix_gl + self.alpha_mix_loc)
        term4 = (self.n_d_v_loc_z[d][s:s+self.T,].T + self.alpha_loc) / (self.n_d_v_loc[d][s:s+self.T] + self.K_loc*self.alpha_loc)
        score_loc = term1 * np.reshape(term2, (3,1)) * np.reshape(term3, (3,1)) * term4.T
        score_loc = np.asarray(score_loc).reshape(-1)
        return np.concatenate([score_glob, score_loc])


    def update(self):
        """ Perform 1 step of Gibbs sampling """
        self.updates = 0
        self.log_lik = 0.0

        for d,doc in enumerate(self.docs):
            for i,w in enumerate(doc):
                s = self.sent_idx[d][i]

                v = self.v_d_s_n[d][i]
                r = self.r_d_s_n[d][i]
                z = self.z_d_s_n[d][i]

                word = self.word_idx[w]

                # remove this word from the counts
                if r == 0:
                    self.n_gl_z_w[z][word]      -= 1
                    self.n_gl_z[z]              -= 1
                    self.n_d_v_gl[d][s+v]       -= 1
                    self.n_d_gl_z[d][z]         -= 1
                    self.n_d_gl[d]              -= 1
                else:
                    self.n_loc_z_w[z][word]     -= 1
                    self.n_loc_z[z]             -= 1
                    self.n_d_v_loc[d][s+v]      -= 1
                    self.n_d_v_loc_z[d][s+v][z] -= 1

                self.n_d_s_v[d][s][v]       -= 1
                self.n_d_s[d][s]            -= 1
                self.n_d_v[d][s+v]          -= 1


                # get distribution of topics
                p_v_r_z = self.p_v_r_z(d, i, w)

                prob = p_v_r_z / p_v_r_z.sum()
                new_p_v_r_z = np.random.multinomial(1, prob)
                new_p_v_r_z_idx = new_p_v_r_z.argmax()
                prob = prob[new_p_v_r_z_idx]
                new_v, new_r, new_z = self.label_v_r_z[new_p_v_r_z_idx]

                # update
                if new_r == 0:
                    self.n_gl_z_w[new_z][word]          += 1
                    self.n_gl_z[new_z]                  += 1
                    self.n_d_v_gl[d][s+new_v]           += 1
                    self.n_d_gl_z[d][new_z]             += 1
                    self.n_d_gl[d]                      += 1
                elif new_r == 1:
                    self.n_loc_z_w[new_z][word]         += 1
                    self.n_loc_z[new_z]                 += 1
                    self.n_d_v_loc[d][s+new_v]          += 1
                    self.n_d_v_loc_z[d][s+new_v][new_z] += 1

                self.n_d_s_v[d][s][new_v]               += 1
                self.n_d_s[d][s]                        += 1
                self.n_d_v[d][s+new_v]                  += 1
                
                self.v_d_s_n[d][i] = new_v
                self.r_d_s_n[d][i] = new_r
                self.z_d_s_n[d][i] = new_z

                self.log_lik += np.log(prob)
                if new_z != z:
                    self.updates += 1
