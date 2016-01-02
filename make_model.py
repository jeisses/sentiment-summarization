from mg_lda import MgLda
import data
import time
import cPickle as pickle
import numpy as np
import os, datetime
import matplotlib.pyplot as plt

times   = []

iterations      = 20
n_local_topics  = 10
n_global_topics = 25

def train_model():
    updates = []
    log_lik = []

    # Train model
    print "Parsing dir.."
    docs, vocab, word_idx, sent_idx, sentences = data.parse_dir("./data/small/")
    l = MgLda(n_local_topics,
              n_global_topics,
              docs, vocab, word_idx, sentences, sent_idx)
    print "Done! Running %d iterations..."%iterations
    for i in range(0, iterations):
        start = time.time()
        l.update()
        duration = time.time() - start
        updates.append(l.updates)
        log_lik.append(l.log_lik)
        times.append(duration)
        print "Iteration %d, Duration: %f"%(i, duration)

    # Save model and data
    dir = os.path.join("models/", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(dir)
    with open(dir+'/mglda_model_2.pkl', 'wb') as output:
        pickle.dump(l, output, -1)

    updates = np.asarray(updates)
    log_lik = np.asarray(log_lik)
    np.savetxt(dir+'/updates.txt', updates)
    np.savetxt(dir+'/log_lik.txt', log_lik)
    np.savetxt(dir+'/times.txt', np.asarray(times))
    print "Model and data saved!"

    # Add some plots
    print "Adding some plots.."
    x = np.arange(0, iterations)
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(x, log_lik/100000)
    plt.xlabel("Iterations")
    plt.ylabel("Log likelihood x 10^5")
    plt.title("MG-LDA training")
    plt.grid(True)
    fig.savefig(dir + "/log_lik.png")
    plt.close(fig)

    # Plot log updates 
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(x, updates)
    plt.xlabel("Iterations")
    plt.ylabel("Updates")
    plt.title("MG-LDA training")
    plt.grid(True)
    fig.savefig(dir + "/updates.png")
    plt.close(fig)

    # Return the model
    return l

def load_last_model():
    last_model_dir = "models/" + os.listdir("models/")[-1] + "/"
    last_model = last_model_dir + "mglda_model_2.pkl"
    updates = np.loadtxt(last_model_dir + 'updates.txt')
    log_lik = np.loadtxt(last_model_dir + 'log_lik.txt')

    print "Loading %s"%last_model
    l = None
    with open(last_model, 'rb') as input:
        l = pickle.load(input)
    return l,updates,log_lik

# load a created model
l,updates,log_lik = load_last_model()
print "Loaded latest model. Local topics are:"
l.top_words(10, "loc")


print "Loaded latest model. Global topics are:"
l.top_words(10, "glob")

