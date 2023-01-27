from nanoshakes_model import NanoShakes
import torch
from lptimer import LP_Timer as Timer

# hyper parameters
input_size = 32
batch_size = 16
layer_count = 4
head_count = 4
embd_size = 16 * 4
dropout = 0.0

use_high_values = False
evaluateDetailedLoss = True
eval_interval = 100
learning_rate = 1e-3
epoch_count = 5000
eval_average_loss_n = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

if use_high_values:
    input_size = 256
    batch_size = 64
    eval_interval = 500
    learning_rate = 3e-4
    embd_size = 384
    head_count = 6
    layer_count = 6
    dropout = 0.2

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
                   layer_count=layer_count, head_count=head_count, dropout=dropout, device=device)

model.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
timer = Timer().start()
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
        if evaluateDetailedLoss:
            losses = calcAverageError(model)
            print(f"""Epoch: {epoch}\t\tTime {timer.elapsed()}
            \t\tTrainloss: {losses['train']:.4f}\t\tValloss: {losses['val']:.4f}""")
        else:
            print(f"Epoch: {epoch}\t\tTime {timer.elapsed()}")

print('\n')
training_time = timer.stop()
print(f'Training Time: {training_time}')
print('\n')

timer.restart()
# just a tensor to start with and configuring the device its on
start = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(start, 2000)[0].tolist()))
print('\n')
print(f'Sequence Generation Time: {timer.stop()}')

# parameters of the model won't be save by this
# evaluate.py for further information
torch.save(model.state_dict(), "trained.pth")
