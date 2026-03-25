# Assignment-2: NLP and Sequence Modeling

This repository contains implementations for two core NLP tasks:

1. Learning Word Embeddings using CBOW & Skip-gram
2. Character-Level Name Generation using RNN variants

The focus of this repository is **implementation from scratch, experimentation, and reproducibility**.

# Problem 1: Word Embeddings (CBOW & Skip-gram)

## Features

* CBOW implemented from scratch (NumPy)
* Skip-gram with Negative Sampling
* Custom preprocessing pipeline
* Cosine similarity for evaluation


##  How to Run (Problem 1)

### Execute Command

```bash
python problem1.py
```
Word Cloud of most occuring words 
<img width="1000" height="500" alt="wordcloud" src="https://github.com/user-attachments/assets/639373e5-ffbc-4900-825f-f8534dcedcbc" />

## Evalaution result of our word embedding in finding nearest neighbour
Target word "research "    
Top 5 Neighbors (SGNS):- 'envisions', 'fellows', 'collaborative', 'thrust', 'niche'


Top 5 Neighbors CBOW:- industry', 'outreach', 'collaborations', 'academia', 'centre'




# Problem 2: Name Generation using RNNs

## Features

* Vanilla RNN (from scratch)
* Bidirectional LSTM (custom implementation)
* Attention-based RNN
* Character-level generation
* Diversity & Novelty evaluation

---

## Dataset

* File: `TrainingNames.txt`
* Contains ~1000 Indian names

---

## How to Run (Problem 2)

### Execute Command

```bash
python problem2.py
```
## Output

After training, the script will print:

* Training loss
* Diversity score
* Novelty rate
* Sample generated names
<img width="1524" height="568" alt="image" src="https://github.com/user-attachments/assets/f704e481-5ee1-4ac6-8416-6ac87af56ff0" />


---

# ⚙️ Requirements

* Python 3.8+
* NumPy
* PyTorch

---



