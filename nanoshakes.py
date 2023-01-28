import torch
import torch.nn as nn
from torch.nn import functional as F

"""
  NanoShakes
    Parameters:
    x: Tensor of shape batch_size x input_size
"""


class NanoShakes(nn.Module):
    def __init__(self, vocab_size, input_size, embd_size, layer_count, head_count, dropout, device):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.word_embedding = nn.Embedding(vocab_size, embd_size)
        self.pos_embedding = nn.Embedding(input_size, embd_size)
        self.norm1 = nn.LayerNorm(embd_size)
        self.transformer = nn.Sequential(
            *[TransformerBlock(embd_size, head_count, input_size, dropout) for _ in range(layer_count)])
        self.embd2vocab = nn.Linear(embd_size, vocab_size)

    def forward(self, x, targets=None):
        we = self.word_embedding(x)  # batch_size x input_size x embd_size
        # doesn't seem very intuitive
        # input positions are encoded to an additively used embd_size'ed vector
        # all i understand is that the positional information is there in some kind
        # of way, but how it could be utilized in this form is totally unclear to me
        # but well, adam will sort it out
        # need to grab the current input size with x.shape[1]
        # because it doesn't have to be the maximum of input_size
        # batch_size x input_size x embd_size
        wp = self.pos_embedding(torch.arange(x.shape[1], device=self.device))
        output = we + wp
        output = self.transformer(output)
        output = self.norm1(output)
        output = self.embd2vocab(output)

        if targets is None:
            loss = None
        else:
            B, T, C = output.shape
            output = output.view(B*T, C)
            targets = targets.view(B*T)
            # output and targets have different shapes
            # targets is just a vector of the token ids
            # output is a matrix where every line shows a probability for every token
            # in the vocab
            loss = F.cross_entropy(output, targets)

        return output, loss

    def generate(self, x, count):
        for _ in range(count):
            # as new text keeps extending in this loop it needs to be cropped to
            # the allowed input_size
            trail = x[:, -self.input_size:]
            out, loss = self(trail)
            out = out[:, -1, :]  # batch_size x vocab_size
            dist = F.softmax(out, dim=-1)
            next = torch.multinomial(dist, num_samples=1)
            x = torch.cat((x, next), dim=1)
        return x


"""
Transformer
  Parameters:
  x: Tensor of shape batch_size x input_size x embd_size
"""


class TransformerBlock(nn.Module):
    def __init__(self, embd_size, head_count, input_size, dropout):
        super().__init__()
        head_size = embd_size // head_count
        self.head = MultiHead(embd_size, input_size, dropout, head_count)
        self.proj = nn.Linear(embd_size, embd_size)
        self.drop = nn.Dropout(dropout)
        self.lastFF = FF_Transformer(embd_size, dropout)
        self.norm1 = nn.LayerNorm(embd_size)
        self.norm2 = nn.LayerNorm(embd_size)

    def forward(self, x):
        out = self.norm1(x)
        out = self.head(out)
        out = self.drop(self.proj(out))
        # This is additive because it's a residual connection also called skipconnection
        # idea is kinda branching the calculation and bringing it back together
        x = x + out
        x = x + self.lastFF(self.norm2(x))
        return x


class MultiHead(nn.Module):
    def __init__(self, embd_size, input_size, dropout, head_count):
        super().__init__()
        # instead of multiplying q,k,v separately
        # we can do the equivalent by using a bigger matrix
        # and split the result afterwards        
        self.qkv = nn.Linear(embd_size, 3 * embd_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(input_size, input_size)).view(1, 1, input_size, input_size))
        self.dropout = nn.Dropout(dropout)
        self.head_count = head_count

    def forward(self, x):
        # batch_size x current_input_size x embd_size
        B, I, E = x.shape
        # q, k, v are of shape B x I x E
        q, k, v = self.qkv(x).split(E, 2)  
        # splitting matrices into chunks aka heads for performance
        q = q.view(B, I, self.head_count,  E //
                   self.head_count).permute(0, 2, 1, 3)
        k = k.view(B, I, self.head_count,  E //
                   self.head_count).permute(0, 2, 1, 3)
        v = v.view(B, I, self.head_count,  E //
                   self.head_count).permute(0, 2, 1, 3)

        # so far splitting matrices didn't change results                
        # but multiplying the heads separately
        # and concatenating them later 
        # isn't a equivalent operation anymore
        x = q @ k.transpose(-2, -1)  # batch_size x input_size x input_size
        x = x*x.shape[-1]**-0.5  # scale result 1/sqrt(d_k) in the paper
        x = x.masked_fill(self.tril[:, :, :I, :I] == 0, float('-inf'))
        x = F.softmax(x, dim=-1)
        x = self.dropout(x)
        # batch_size x input_size x embd_size
        x = x @ v
        # concatenating head results again
        x = x.permute(0, 2, 1, 3).reshape(B, I, E)
        return x


class FF_Transformer(nn.Module):
    def __init__(self, embd_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embd_size, 4*embd_size),
            nn.ReLU(),
            nn.Linear(4*embd_size, embd_size),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)
