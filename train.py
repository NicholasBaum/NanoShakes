from nanoshakes_model import NanoShakes
from ng_gpt_model import BigramLanguageModel
import time
from datetime import timedelta
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyper parameters
input_size = 32
batch_size = 16
layer_count = 4
head_count = 4
embd_size = 16 * 4
dropout = 0.0

eval_interval = 100
learning_rate = 1e-3
epoch_count = 50
eval_average_loss_n = 200
device = 'cpu' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# load data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocab = sorted(list(set(text)))
vocab_size = len(vocab)

# encoding
char2ind = {ch: i for i, ch in enumerate(vocab)}
ind2char = {i: ch for i, ch in enumerate(vocab)}


def encode(x): return [char2ind[c] for c in x]
def decode(x): return ''.join(ind2char[i] for i in x)


# training data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def getBatch(split):
    data = train_data if split == 'train' else val_data
    # create random starting points
    ind = torch.randint(len(data)-input_size, (batch_size,))
    x = torch.stack([data[i:i+input_size] for i in ind])
    # y is x shifted one position to the right, so y[i] is the next token and the correct guess for x[i] in a sentence
    y = torch.stack([data[i+1:i+input_size+1] for i in ind])
    x, y = x.to(device), y.to(device)
    return x, y


# training

# better error estimation by averaging over differnt sets
@torch.no_grad()
def calcAverageError(model):
    model.eval()
    losses = {key: 0.0 for key in ['train', 'val']}
    for key in losses:
        for x in range(eval_average_loss_n):
            xb, yb = getBatch(key)
            _, loss = model(xb, yb)
            losses[key] += loss.item()
    losses = {key: val/eval_average_loss_n for key, val in losses.items()}
    model.train()
    return losses


print("Using " + device)
model = NanoShakes(vocab_size=vocab_size, input_size=input_size, embd_size=embd_size,
                   layer_count=layer_count, head_count=head_count, dropout=dropout)
model = BigramLanguageModel(vocab_size=vocab_size, n_embd=embd_size, block_size=input_size,
                            n_layer=layer_count, n_head=head_count, dropout=dropout)

model.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
start_time = time.time()
for epoch in range(epoch_count):
    xBatch, yBatch = getBatch('train')
    # forward
    output, loss = model(xBatch, yBatch)
    # backwards
    # resets the gradients, e.g. in RNNs you wouldn't necessarily do this
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    if epoch % eval_interval == 0 or epoch == epoch_count-1:
        losses = calcAverageError(model)
        print(
            f"Epoch: {epoch}\t\tTrainloss: {losses['train']:.4f}\t\tValloss: {losses['val']:.4f}")
print('\n')
# just a tensor to start with and configuring the device its on
start = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(start, 2000)[0].tolist()))
print('\n')
print(str(timedelta(seconds=time.time()-start_time)))