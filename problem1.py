#here we define and import our libraries  
import os
import re
import fitz
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from wordcloud import WordCloud

class IITJCorpus:
    def __init__(self, pdf_path, min_freq=2):
        self.pdf_path = pdf_path
        self.min_freq = min_freq
        # here we set up stopwords
        self.stop_words = set(['the', 'of', 'es','ci','pkdd', 'else','and', 'in', 'to', 'for', 'is', 'on', 'at', 'by', 'with', 'from', 'as', 'an', 'it', 'that', 'this', 'shall', 'be', 'will', 'a', 'are', 'was', 'were', 'had', 'has', 'can', 'such', 'other', 'which','him','her'])

    def clean_and_prepare(self):
        fitz.TOOLS.mupdf_display_errors(False) 
        raw_text = ""
        
        if not os.path.exists(self.pdf_path):
            print(f"Directory {self.pdf_path} not found!")
            return

        #here we get pdf files
        files = [f for f in os.listdir(self.pdf_path) if f.endswith(".pdf")]
        for file in files:
            try:
                #here we read pdf content
                doc = fitz.open(os.path.join(self.pdf_path, file))
                for page in doc: 
                    raw_text += page.get_text() + " "
                doc.close()
            except: 
                continue
        
        #we convert text to lower case all
        text = raw_text.lower()
        
        text = re.sub(r'\bb[\.\-]tech\b', 'btech', text)
        text = re.sub(r'\bm[\.\-]tech\b', 'mtech', text)
        text = re.sub(r'\bph[\.\-]d\b', 'phd', text)
        text = re.sub(r'\bug\b', 'ug', text)
        text = re.sub(r'\bpg\b', 'pg', text)
        #remove marks
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        #split words
        words = text.split()
        #remove stopwords
        self.tokens = [w for w in words if w not in self.stop_words and len(w) > 1]
        counts = Counter(self.tokens)
        #filter frequency
        self.vocab = {w: c for w, c in counts.items() if c >= self.min_freq}
        
        if not self.vocab:
            print("Warning: Vocabulary is empty. Check PDF content.")
            return

        #here we map word to index
        self.word2idx = {w: i for i, w in enumerate(self.vocab.keys())}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        
        #here we save to text
        with open("cleaned_corpus.txt", "w") as f: 
            f.write(" ".join(self.tokens))
        
        print(f"Stats: Documents: {len(files)} | Tokens: {len(self.tokens)} | Vocab: {len(self.vocab)}")
        
        #here we make word cloud
        wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(self.tokens))
        plt.figure(figsize=(10, 5)); plt.imshow(wc); plt.axis("off"); plt.savefig("wordcloud.png")

class Word2VecScratch:
    def __init__(self, vocab_size, emb_dim=128):
        self.v_size = vocab_size
        self.dim = emb_dim
        #set random weights
        self.W1 = np.random.uniform(-0.1, 0.1, (vocab_size, emb_dim))
        self.W2 = np.random.uniform(-0.1, 0.1, (vocab_size, emb_dim))

    def train_sgns(self, corpus_idxs, corpus_obj, epochs=15, lr=0.02, window=5, k=10):
        print("Training SGNS...", flush=True)
        #make table for neg sampling
        pow_counts = np.array([corpus_obj.vocab[corpus_obj.idx2word[i]] for i in range(self.v_size)])**0.75
        table = np.repeat(np.arange(self.v_size), (pow_counts / pow_counts.sum() * 1e6).astype(int))

        for epoch in range(epochs):
            curr_lr = lr * (1 - epoch/epochs)
            for i, target in enumerate(corpus_idxs):
                #here we do sub sampling
                z = corpus_obj.vocab[corpus_obj.idx2word[target]] / len(corpus_obj.tokens)
                if np.random.random() > (np.sqrt(z/0.001) + 1) * (0.001/z): continue

                #here get context
                start, end = max(0, i-window), min(len(corpus_idxs), i+window+1)
                for j in range(start, end):
                    if i == j: continue
                    ctx, negs = corpus_idxs[j], table[np.random.randint(0, len(table), k)]
                    idx, labels = np.append(ctx, negs), np.append(1, np.zeros(k))
                    
                    #here we calculate loss
                    probs = 1 / (1 + np.exp(-np.clip(np.dot(self.W2[idx], self.W1[target]), -10, 10)))
                    err = probs - labels
                    
                    #update weights
                    grad_w1 = np.dot(err, self.W2[idx])
                    self.W2[idx] -= curr_lr * np.outer(err, self.W1[target])
                    self.W1[target] -= curr_lr * grad_w1
            print(f"SGNS Epoch {epoch+1} complete", flush=True)

    def train_cbow(self, corpus_idxs, epochs=15, lr=0.02, window=5):
        print("Training CBOW...", flush=True)
        for epoch in range(epochs):
            curr_lr = lr * (1 - epoch/epochs)
            for i in range(window, len(corpus_idxs)-window):
                target = corpus_idxs[i]
                #here get context words
                ctx_idx = [corpus_idxs[j] for j in range(i-window, i+window+1) if i != j]
                
                h = np.mean(self.W1[ctx_idx], axis=0)
                
                #predict target
                dots = np.dot(self.W2, h)
                preds = np.exp(dots - np.max(dots)) / np.exp(dots - np.max(dots)).sum()
                err = preds.copy(); err[target] -= 1
                
                #update vectors
                self.W2 -= curr_lr * np.outer(err, h)
                self.W1[ctx_idx] -= curr_lr * (np.dot(err, self.W2) / len(ctx_idx))
            print(f"CBOW Epoch {epoch+1} complete", flush=True)

class Analyst:
    def __init__(self, model, corpus, label):
        #here we normalize weights
        self.weights = model.W1 / (np.linalg.norm(model.W1, axis=1, keepdims=True) + 1e-9)
        self.w2i, self.i2w, self.label = corpus.word2idx, corpus.idx2word, label

    def get_neighbors(self, word, k=5):
        #here we find closest words
        if word not in self.w2i: return []
        sims = np.dot(self.weights, self.weights[self.w2i[word]])
        return [self.i2w[i] for i in np.argsort(sims)[-k-1:-1][::-1]]

    def analogy(self, a, b, c):
        #here we solve math analogy
        if not all(w in self.w2i for w in [a, b, c]): return "N/A"
        vec = self.weights[self.w2i[b]] - self.weights[self.w2i[a]] + self.weights[self.w2i[c]]
        sims = np.dot(self.weights, vec)
        return [self.i2w[i] for i in np.argsort(sims) if self.i2w[i] not in [a, b, c]][-1]

    def plot_dense(self, targets):
        selected = set(targets)
        for w in targets: 
            neighbors = self.get_neighbors(w, k=15)
            if neighbors: selected.update(neighbors)
        
        #here we add add common words
        common = [w for w, _ in Counter(corpus.vocab).most_common(20)]
        selected.update(common)
        
        final_list = [w for w in selected if w in self.w2i]
        if not final_list: return
        data = self.weights[[self.w2i[w] for w in final_list]]

        #reduce and plot
        for name, algo in [("PCA", PCA(n_components=2)), ("TSNE", TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=min(30, len(final_list)-1)))]:
            coords = algo.fit_transform(data)
            plt.figure(figsize=(12, 10))
            plt.scatter(coords[:, 0], coords[:, 1], c='magenta', alpha=0.5)
            for i, txt in enumerate(final_list):
                plt.annotate(txt, (coords[i,0], coords[i,1]), fontsize=8, color='red' if txt in targets else 'blue')
            plt.title(f"{name} Clusters - {self.label}")
            plt.savefig(f"{name}_{self.label}.png")
            print(f"Saved {name}_{self.label}.png")

#this is the start script
corpus = IITJCorpus("./data")
corpus.clean_and_prepare()

if hasattr(corpus, 'tokens'):
    c_idxs = [corpus.word2idx[w] for w in corpus.tokens if w in corpus.word2idx]

    #Here we run sgns
    sg_model = Word2VecScratch(len(corpus.vocab))
    sg_model.train_sgns(c_idxs, corpus)

    #Here we run cbow
    cb_model = Word2VecScratch(len(corpus.vocab))
    cb_model.train_cbow(c_idxs)

    #Here we show results
    for label, m in [("SGNS", sg_model), ("CBOW", cb_model)]:
        ev = Analyst(m, corpus, label)
        print(f"\n--- {label} Results ---")
        for w in ['research', 'student', 'phd', 'exam']:
            print(f"Neighbors for {w}: {ev.get_neighbors(w)}")
        print(f"Analogy UG:BTech :: PG: {ev.analogy('ug', 'btech', 'pg')}")
        ev.plot_dense(['research', 'student', 'phd', 'exam', 'faculty', 'senate', 'ug', 'pg'])