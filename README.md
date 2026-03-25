# Assignment-2: NLP and Sequence Modeling

This repository contains implementations for two core NLP tasks:

1. Learning Word Embeddings (CBOW & Skip-gram)
2. Character-Level Name Generation using RNN variants

The focus of this repository is **implementation from scratch, experimentation, and reproducibility**.

---

#Repository Structure

```
Assignment-2/
│── problem1_word2vec/
│   ├── cbow.py
│   ├── skipgram.py
│   ├── preprocess.py
│   └── data/
│
│── problem2_name_generation/
│   ├── train.py
│   ├── models.py
│   ├── utils.py
│   └── TrainingNames.txt
│
│── README.md
```

---

# Problem 1: Word Embeddings (CBOW & Skip-gram)

## Features

* CBOW implemented from scratch (NumPy)
* Skip-gram with Negative Sampling
* Custom preprocessing pipeline
* Cosine similarity for evaluation

---

##  How to Run (Problem 1)

### Step 1: Navigate to directory

```bash
cd problem1_word2vec
```

### Step 2: Run preprocessing

```bash
python preprocess.py
```

### Step 3: Train CBOW model

```bash
python cbow.py
```

### Step 4: Train Skip-gram model

```bash
python skipgram.py
```

---

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

### Step 1: Navigate to directory

```bash
cd problem2_name_generation
```

### Step 2: Install dependencies

```bash
pip install torch numpy
```

### Step 3: Run training

```bash
python train.py
```

---

## Output

After training, the script will print:

* Training loss
* Diversity score
* Novelty rate
* Sample generated names

---

# ⚙️ Requirements

* Python 3.8+
* NumPy
* PyTorch

---

# Notes

* All models are implemented from scratch (no high-level libraries like gensim)
* Ensure dataset files are placed correctly before running
* GPU will be used automatically if available

---

# Author

Coursework submission for NLP and Deep Learning
