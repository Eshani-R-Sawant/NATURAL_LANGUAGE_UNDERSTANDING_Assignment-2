# -*- coding: utf-8 -*-


#Here we import all the neccesary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import random
import string
import unicodedata
import os

# Heere we define allowed characters  that are letters,space ,punctuation
ALL_LETTERS = string.ascii_letters + " .'-"
#Here total number of characters +1 is added  for EOS token
N_LETTERS = len(ALL_LETTERS) + 1
# Index for End-Of-Sequence token is defined here
EOS_IDX = N_LETTERS - 1
#Hyperparameters of the model
HIDDEN_SIZE = 128
LR = 0.0003
ITERS = 30000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# here we convert unicode string to plain ASCII
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn' and c in ALL_LETTERS)

# here load names from file or use default samples
def load_data(filename="TrainingNames.txt"):
    if not os.path.exists(filename):
        return ["Aarav Sharma", "Riya Gupta", "Arjun Reddy", "Sneha Iyer"]
    with open(filename, 'r', encoding='utf-8') as f:
        return [unicode_to_ascii(line.strip()) for line in f if len(line.strip()) > 1]

## here we convert name into one-hot encoded tensor
def name_to_tensor(name):
    tensor = torch.zeros(len(name), 1, N_LETTERS).to(DEVICE)
    for i, char in enumerate(name):
        idx = ALL_LETTERS.find(char)
        if idx != -1:
            tensor[i][0][idx] = 1
    return tensor

# here we create target indices for next character prediction
def target_to_tensor(name):
    idxs = [ALL_LETTERS.find(name[i]) for i in range(1, len(name))]
    idxs.append(EOS_IDX)
    return torch.LongTensor(idxs).to(DEVICE)

# Here we count total trainable parameters in model
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



#Here we define from scratch the RNN model
class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input_step, hidden):
        combined = torch.cat((input_step, hidden), 1)
        hidden = torch.tanh(self.i2h(combined))
        output = self.i2o(combined)
        return output, hidden


# HEre we define from scartch BLSTM (Encoder + Decoder)  model
class BLSTM_Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # HEre we define forward and backward LSTM gates
        self.f_gate = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        self.b_gate = nn.Linear(input_size + hidden_size, 4 * hidden_size)


        self.dec_gate = nn.Linear(input_size + hidden_size, 4 * hidden_size)

        self.out = nn.Linear(hidden_size, output_size)
    # Here we define  Single LSTM step
    def lstm_step(self, layer, x, h, c):
        gates = layer(torch.cat((x, h), 1))
        i, f, o, g = gates.chunk(4, 1)

        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)

        return h, c
    #Here Encode input using bidirectional LSTM
    def encode(self, it):
        fh = torch.zeros(1, self.hidden_size).to(DEVICE)
        fc = torch.zeros(1, self.hidden_size).to(DEVICE)

        bh = torch.zeros(1, self.hidden_size).to(DEVICE)
        bc = torch.zeros(1, self.hidden_size).to(DEVICE)


        for i in range(it.size(0)):
            fh, fc = self.lstm_step(self.f_gate, it[i], fh, fc)


        for i in reversed(range(it.size(0))):
            bh, bc = self.lstm_step(self.b_gate, it[i], bh, bc)

        return (fh + bh) / 2  # combine
    # Here we define Decode sequence
    def forward(self, it):
        context = self.encode(it)

        h = context
        c = torch.zeros_like(h)

        outputs = []
        for i in range(it.size(0)):
            h, c = self.lstm_step(self.dec_gate, it[i], h, c)
            outputs.append(self.out(h))

        return torch.stack(outputs)


#Here we define Attention RNN from scartch here
class AttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Here we Compute current hidden state

        self.rnn = nn.Linear(input_size + hidden_size, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_step, hidden, history):
        h = torch.tanh(self.rnn(torch.cat((input_step, hidden), 1)))

        if history.size(0) == 0:
            context = torch.zeros(1, self.hidden_size).to(DEVICE)
        else:
            h_rep = h.repeat(history.size(0), 1)
            scores = self.attn(torch.cat((h_rep, history), 1))
            weights = torch.softmax(scores, dim=0)
            context = torch.sum(weights * history, dim=0).unsqueeze(0)

        output = self.out(torch.cat((h, context), 1))
        return output, h


# Here we define code to generate new name from trained model
def generate(model, start_char, temp=0.7):
    model.eval()
    result = start_char
    # Here we initialize input vector
    inp = torch.zeros(1, N_LETTERS).to(DEVICE)
    idx = ALL_LETTERS.find(start_char)
    if idx != -1:
        inp[0][idx] = 1
    # Here we initialize input vector
    h = torch.zeros(1, HIDDEN_SIZE).to(DEVICE)
    hist = torch.zeros(0, HIDDEN_SIZE).to(DEVICE)

    hf = torch.zeros(1, HIDDEN_SIZE).to(DEVICE)
    cf = torch.zeros(1, HIDDEN_SIZE).to(DEVICE)

    with torch.no_grad():
        for _ in range(30):
            #Here we define different forward logic for each model
            if isinstance(model, BLSTM_Generator):
                h, cf = model.lstm_step(model.dec_gate, inp, hf, cf)
                logits = model.out(h)
                hf = h

            elif isinstance(model, AttentionRNN):
                logits, h = model(inp, h, hist)
                hist = torch.cat((hist, h), 0)

            else:
                logits, h = model(inp, h)


            if len(result) > 5:
                space_idx = ALL_LETTERS.find(' ')
                logits[0, space_idx] += 2.5

            # here we define code to avoid repetition
            if len(result) > 2 and result[-1] == result[-2]:
                ridx = ALL_LETTERS.find(result[-1])
                logits[0, ridx] -= 2.0


            for v in "aeiouAEIOU":
                vidx = ALL_LETTERS.find(v)
                logits[0, vidx] += 0.3

            probs = torch.softmax(logits / temp, dim=1)
            top_i = torch.multinomial(probs, 1).item()

            if top_i == EOS_IDX:
                if len(result) < 6:
                    continue
                else:
                    break

            char = ALL_LETTERS[top_i]
            result += char

            inp.fill_(0)
            inp[0][top_i] = 1

    return result


#HEre we define code for Training nad evaluation
def run_task():
    names = load_data()
    train_set = set(names)
    # Here we  intialize all  models

    models = {
        "vanilla": VanillaRNN(N_LETTERS, HIDDEN_SIZE, N_LETTERS).to(DEVICE),
        "blstm": BLSTM_Generator(N_LETTERS, HIDDEN_SIZE, N_LETTERS).to(DEVICE),
        "attention": AttentionRNN(N_LETTERS, HIDDEN_SIZE, N_LETTERS).to(DEVICE)
    }
    # Here we define code to train each model

    for name, model in models.items():
        print(f"\n--- Training {name.upper()} ---")
        print(f"Parameters: {count_params(model)}")

        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        for i in range(1, ITERS + 1):
            target_name = random.choice(names)

            it = name_to_tensor(target_name)
            tt = target_to_tensor(target_name)

            optimizer.zero_grad()

            if name == "blstm":
                out_seq = model(it).squeeze(1)
                loss = 0
                for j in range(len(tt)):
                    loss += criterion(out_seq[j].unsqueeze(0), tt[j].unsqueeze(0))
                loss /= len(tt)

            else:
                loss = 0
                h = torch.zeros(1, HIDDEN_SIZE).to(DEVICE)
                hist = torch.zeros(0, HIDDEN_SIZE).to(DEVICE)

                for j in range(it.size(0)):
                    if name == "attention":
                        out, h = model(it[j], h, hist)
                        hist = torch.cat((hist, h), 0)
                    else:
                        out, h = model(it[j], h)

                    loss += criterion(out, tt[j].unsqueeze(0))

                loss /= it.size(0)
            #Here we deine the code of backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            if i % 10000 == 0:
                print(f"Iteration {i} | Loss: {loss.item():.4f}")

        # HEre we define the code for evaluation
        generated = [generate(model, random.choice(string.ascii_uppercase)) for _ in range(1000)]
        unique_names = set(generated)
        novel_names = [n for n in unique_names if n not in train_set]

        print(f"\nResults for {name.upper()}:")
        print(f"Diversity: {len(unique_names)/1000:.2%}")
        print(f"Novelty Rate: {len(novel_names)/len(unique_names):.2%}")
        print(f"12 Samples: {', '.join(generated[:12])}")


if __name__ == "__main__":
    run_task()

