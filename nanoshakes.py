# -*- coding: utf-8 -*-
"""NanoShakes.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ib2Byi7edeZ0ABwcAeETtg_YWIZ55QHO
"""

# get data, the exclamation mark tells python to run the command by your operating system not by the runtime/shell
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

import time
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
epoch_count = 5000
eval_average_loss_n = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# load data
with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

vocab = sorted(list(set(text)))
vocab_size = len(vocab)

# encoding
char2ind = {ch:i for i,ch in enumerate(vocab)}
ind2char = {i:ch for i,ch in enumerate(vocab)}

encode = lambda x: [char2ind[c] for c in x]
decode = lambda x: ''.join(ind2char[i] for i in x)

# training data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def getBatch(split):
  data = train_data if split=='train' else val_data
  # create random starting points
  ind = torch.randint(len(data)-input_size, (batch_size,)) 
  x = torch.stack([data[i:i+input_size] for i in ind])
  # y is x shifted one position to the right, so y[i] is the next token and the correct guess for x[i] in a sentence
  y = torch.stack([data[i+1:i+input_size+1] for i in ind])
  x, y = x.to(device), y.to(device)
  return x,y

"""
  NanoShakes
    Parameters:
    x: Tensor of shape batch_size x input_size
"""
class NanoShakes(nn.Module):
  def __init__(self):
    super().__init__()
    self.word_embedding = nn.Embedding(vocab_size, embd_size)
    self.pos_embedding = nn.Embedding(input_size, embd_size)
    self.norm1 = nn.LayerNorm(embd_size);
    self.transformer = nn.Sequential(*[TransformerBlock(embd_size) for _ in range(layer_count)])
    self.embd2vocab = nn.Linear(embd_size, vocab_size)
  
  def forward(self, x, targets = None):    
    we = self.word_embedding(x) # batch_size x input_size x embd_size    
    # doesn't seem very intuitive
    # input positions are encoded to an additively used embd_size'ed vector
    # all i understand is that the positional information is there in some kind
    # of way, but how it could be utilized in this form is totally unclear to me
    # but well, adam will sort it out
    wp = self.pos_embedding(torch.arange(x.shape[1], device=device)) # batch_size x input_size x embd_size    # need to use x.shape[1] because sentences could be shorter thatn input_size
    output = we + wp
    output = self.transformer(output)    
    output = self.norm1(output)
    output = self.embd2vocab(output) 
    
    if targets == None:
      loss = None
    else:
      B,T,C = output.shape
      output = output.view(B*T,C)      
      targets = targets.view(B*T)
      # output and targets have different shapes
      # targets is just a vector of the token ids
      # output is a matrix where very line shows a probability for every token 
      # in the vocab
      loss = F.cross_entropy(output, targets)

    return output, loss

  def generate(self, x, count):
    for _ in range(count):
      # as new text keeps extending in this loop it needs to be cropped to
      # the allowed input_size      
      trail = x[:, -input_size:]
      out, loss = self(trail)
      out = out[:,-1,:] # batch_size x vocab_size
      dist = F.softmax(out, dim=-1)          
      next = torch.multinomial(dist, num_samples=1)
      x = torch.cat((x,next), dim=1)
    return x

"""
Transformer
  Parameters:
  x: Tensor of shape batch_size x input_size x embd_size
"""  
class TransformerBlock(nn.Module):
  def __init__(self, embd_size):
    super().__init__()
    head_size = embd_size // head_count
    # allows parallel modules when put in brackets
    self.heads = nn.ModuleList([Head(embd_size, head_size) for _ in range(head_count)])
    self.proj = nn.Linear(embd_size, embd_size)
    self.drop = nn.Dropout(dropout)
    self.lastFF = FF_Transformer(embd_size)
    self.norm1 = nn.LayerNorm(embd_size)
    self.norm2 = nn.LayerNorm(embd_size)

  def forward(self, x):
    out = self.norm1(x)
    out = torch.cat([h(out) for h in self.heads], dim=-1)       
    out = self.drop(self.proj(out))
    # TODO: don't know why this is additive
    x = x + out    
    x = x + self.lastFF(self.norm2(x))
    return x

"""
  Head
    Parameters:
    x: Tensor of shape batch_size x input_size x embd_size
"""  
class Head(nn.Module):
  def __init__(self, embd_size, head_size):
    super().__init__()
    self.q = nn.Linear(embd_size, head_size, bias = False)
    self.k = nn.Linear(embd_size, head_size, bias = False)
    self.v = nn.Linear(embd_size, head_size, bias = False)
    self.register_buffer('tril', torch.tril(torch.ones(input_size, input_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    curr_input_n = x.shape[1]
    q = self.q(x)
    k = self.k(x)
    v = self.v(x)  
    x = q @ k.transpose(1, 2) # batch_size x input_size x input_size
    x = x*x.shape[2]**0.5 # scale result 1/sqrt(d_k) in the paper
    x = x.masked_fill(self.tril[:curr_input_n,:curr_input_n]==0,float('-inf'))    
    x = F.softmax(x, dim = 2)
    x = self.dropout(x)       
    x = x @ v # batch_size x input_size x embd_size    
    return x

class FF_Transformer(nn.Module):
  def __init__(self, embd_size):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(embd_size, 4*embd_size),
        nn.ReLU(),
        nn.Linear(4*embd_size, embd_size),
        nn.Dropout(dropout))

  def forward(self, x):
    return self.net(x)

# training

# better error estimation by averaging over differnt sets
@torch.no_grad()
def calcAverageError(model):  
  model.eval()
  losses = {key:0.0 for key in ['train', 'val']}
  for key in losses:    
    for x in range(eval_average_loss_n):
      xb, yb = getBatch(key)
      _, loss = model(xb, yb)
      losses[key] += loss.item()
  losses = {key:val/eval_average_loss_n for key, val in losses.items()}
  model.train()
  return losses

print("Using " + device)
model = NanoShakes()
model.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
opt = torch.optim.AdamW(model.parameters(), lr = learning_rate)
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
  
  if epoch%eval_interval == 0 or epoch==epoch_count-1:
    losses = calcAverageError(model)
    print(f"Epoch: {epoch}\t\tTrainloss: {losses['train']:.4f}\t\tValloss: {losses['val']:.4f}")

print(loss.item())
print(time.time()-start_time)
# just a tensor to start with and configuring the device its on
start = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(start, 2000)[0].tolist()))