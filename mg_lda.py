import string
import numpy as np
import data


class MgLda:

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

        self.v_d_s_n  = []  # window assignment for each word
        self.z_d_s_n  = []  # topic assignment for each word
        self.r_d_s_n  = []  # glob (0) / loc (1) assignment for each word

        self.n_d_s    = []  # nr of words in sentence s
        self.n_d_s_v  = []  # nr of words in sentence s assigned to window v
        self.n_d_v    = []  # nr of words in window v

        self.n_d_v_gl  = []  # nr of words in window v assigned to global topics
        self.n_d_v_loc = []  # nr of words in window v assigned to local topics

        self.n_d_gl_z   = np.zeros((n_docs, self.K_gl))  # times d has been assigned to global topic
        self.n_d_gl     = np.zeros(n_docs)          # times d 

        self.n_d_v_loc_z = []  # times window v has been assigned local topic

        self.n_gl_z_w  = np.zeros((self.K_gl, vocab_size))  # times w has been assigned to global topic
        self.n_loc_z_w = np.zeros((self.K_loc, vocab_size)) # times w has been assigned to local topic

        self.n_gl_z  = np.zeros(self.K_gl)  # times global z has been assigned to any word
        self.n_loc_z = np.zeros(self.K_loc) # times local z has been assigned to any word

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
                self.n_d_s_v[i].append([0 for a in range(0, self.T)])
            self.n_d_v.append([0 for a in range(0, n_windows)])

            self.n_d_v_gl.append([0]*n_windows)
            self.n_d_v_loc.append([0]*n_windows)
            self.n_d_v_loc_z.append([])
            for v in range(n_windows):
                self.n_d_v_loc_z[i].append([0]*self.K_loc)


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
                s -= 1

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

    def top_words(self, n):
        """ Print the top x words for each topic """
        for k in xrange(self.K_loc):
            idcs = np.argpartition(self.z_d_s_n,-n)[-n:]
            words = []
            for w in idcs:
                words.append(self.vocab[w])
            print "Local Topic %d: %s"%(k, words)

    def update(self):
        """ Perform 1 step of Gibbs sampling """
        updated = 0

        for d,doc in enumerate(self.docs):
            for i,w in enumerate(doc):
                s = self.sent_idx[d][i] - 1

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

                W = len(self.vocab)
                # sample a new topic
                p_v_r_z = []
                label_v_r_z = []
                for v_t in range(self.T):
                    # for r == "gl"
                    for z_t in range(self.K_gl):
                        label = [v_t, 0, z_t]
                        label_v_r_z.append(label)
                        # sampling eq as gl
                        term1 = float(self.n_gl_z_w[z_t][word] + self.beta_gl) / (self.n_gl_z[z_t] + W*self.beta_gl)
                        term2 = float(self.n_d_s_v[d][s][v_t] + self.gamma) / (self.n_d_s[d][s] + self.T*self.gamma)
                        term3 = float(self.n_d_v_gl[d][s+v_t] + self.alpha_mix_gl) / (self.n_d_v[d][s+v_t] + self.alpha_mix_gl + self.alpha_mix_loc)
                        term4 = float(self.n_d_gl_z[d][z_t] + self.alpha_gl) / (self.n_d_gl[d] + self.K_gl*self.alpha_gl)
                        score = term1 * term2 * term3 * term4
                        p_v_r_z.append(score)
                        # for r == "loc" 
                    for z_t in range(self.K_loc):
                        label = [v_t, 1, z_t]
                        label_v_r_z.append(label)
                        # sampling eq as loc
                        term1 = float(self.n_loc_z_w[z_t][word] + self.beta_loc) / (self.n_loc_z[z_t] + W*self.beta_loc)
                        term2 = float(self.n_d_s_v[d][s][v_t] + self.gamma) / (self.n_d_s[d][s] + self.T*self.gamma)
                        term3 = float(self.n_d_v_loc[d][s+v_t] + self.alpha_mix_loc) / (self.n_d_v[d][s+v_t] + self.alpha_mix_gl + self.alpha_mix_loc)
                        term4 = float(self.n_d_v_loc_z[d][s+v_t][z_t] + self.alpha_loc) / (self.n_d_v_loc[d][s+v_t] + self.K_loc*self.alpha_loc)
                        score = term1 * term2 * term3 * term4
                        p_v_r_z.append(score)


                np_p_v_r_z = np.array(p_v_r_z)
                new_p_v_r_z_idx = np.random.multinomial(1, np_p_v_r_z / np_p_v_r_z.sum()).argmax()
                new_v, new_r, new_z = label_v_r_z[new_p_v_r_z_idx]
 

                # update
                if new_r == "gl":
                    self.n_gl_z_w[new_z][word]          += 1
                    self.n_gl_z[new_z]                  += 1
                    self.n_d_v_gl[d][s+new_v]           += 1
                    self.n_d_gl_z[d][new_z]             += 1
                    self.n_d_gl[d]                      += 1
                elif new_r == "loc":
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

                if new_z != z:
                    updated += 1
        print "Updated %d"%updated




# EXAMPLE
print "Parsing dir.."
docs, vocab, word_idx, sent_idx, sentences = data.parse_dir("./data/")
#docs, vocab, word_idx, sent_idx, sentences = data.parse_dir("/Users/jeisses/Documents/datasets/nlp/movie/review_polarity/txt_sentoken/all/")
print "Done"
print " ====== "
print "Setting up LDA.."

l = MgLda(10, 50, docs, vocab, word_idx, sentences, sent_idx)