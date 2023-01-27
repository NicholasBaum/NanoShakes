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
        # batch_size x input_size x embd_size    # need to use x.shape[1] because sentences could be shorter thatn input_size
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
            # output is a matrix where very line shows a probability for every token
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


class TransformerBlock2(nn.Module):
    def __init__(self, embd_size, head_count, input_size, dropout):
        super().__init__()
        head_size = embd_size // head_count
        # allows parallel modules when put in brackets
        self.heads = nn.ModuleList(
            [Head(embd_size, head_size, input_size, dropout) for _ in range(head_count)])
        self.proj = nn.Linear(embd_size, embd_size)
        self.drop = nn.Dropout(dropout)
        self.lastFF = FF_Transformer(embd_size, dropout)
        self.norm1 = nn.LayerNorm(embd_size)
        self.norm2 = nn.LayerNorm(embd_size)

    def forward(self, x):
        out = self.norm1(x)
        out = torch.cat([h(out) for h in self.heads], dim=-1)
        out = self.drop(self.proj(out))
        # This is additive because it's a residual connection also called skipconnection
        # idea is kinda branching the calculation and bringing it back together
        x = x + out
        x = x + self.lastFF(self.norm2(x))
        return x


"""
  Head
    Parameters:
    x: Tensor of shape batch_size x input_size x embd_size
"""


class Head(nn.Module):
    def __init__(self, embd_size, head_size, input_size, dropout):
        super().__init__()
        self.q = nn.Linear(embd_size, head_size, bias=False)
        self.k = nn.Linear(embd_size, head_size, bias=False)
        self.v = nn.Linear(embd_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(input_size, input_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        curr_input_n = x.shape[1]
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        x = q @ k.transpose(1, 2)  # batch_size x input_size x input_size
        x = x*x.shape[2]**-0.5  # scale result 1/sqrt(d_k) in the paper
        x = x.masked_fill(
            self.tril[:curr_input_n, :curr_input_n] == 0, float('-inf'))
        x = F.softmax(x, dim=2)
        x = self.dropout(x)
        x = x @ v  # batch_size x input_size x embd_size
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


class TransformerBlock(nn.Module):
    def __init__(self, embd_size, head_count, input_size, dropout):
        super().__init__()
        head_size = embd_size // head_count
        # allows parallel modules when put in brackets
        self.head = MultiHead(embd_size, head_size,
                              input_size, dropout, head_count=head_count)
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


"""
  Head
    Parameters:
    x: Tensor of shape batch_size x input_size x embd_size
"""


class MultiHead2(nn.Module):
    def __init__(self, embd_size, head_size, input_size, dropout, head_count):
        super().__init__()
        self.q = nn.Linear(embd_size, head_size, bias=False)
        self.k = nn.Linear(embd_size, head_size, bias=False)
        self.v = nn.Linear(embd_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(input_size, input_size)).view(1, 1, input_size, input_size))
        self.dropout = nn.Dropout(dropout)
        self.head_count = head_count

    def forward(self, x):
        B, I, E = x.shape  # batch_size x input_size x embd_size
        # I is actually current_input_size
        # possibly less than the max input_size
        # batch_size x head_count x input_size x embd_size
        x = torch.unsqueeze(x, dim=1)
        x = x.repeat(1, self.head_count, 1, 1)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        x = q @ k.transpose(-2, -1)  # batch_size x input_size x input_size
        x = x*x.shape[-1]**-0.5  # scale result 1/sqrt(d_k) in the paper
        x = x.masked_fill(
            self.tril[:, :, :I, :I] == 0, float('-inf'))
        x = F.softmax(x, dim=-1)
        x = self.dropout(x)
        x = x @ v  # batch_size x input_size x embd_size
        x = x.permute(0, 2, 1, 3).reshape(B, I, E)
        return x


class MultiHead(nn.Module):
    def __init__(self, embd_size, head_size, input_size, dropout, head_count):
        super().__init__()
        self.qkv = nn.Linear(embd_size, 3 * embd_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(input_size, input_size)).view(1, 1, input_size, input_size))
        self.dropout = nn.Dropout(dropout)
        self.head_count = head_count

    def forward(self, x):
        B, I, E = x.shape  # batch_size x input_size x embd_size
        # I is actually current_input_size
        # possibly less than the max input_size
        # batch_size x input_size x embd_size
        q, k, v = self.qkv(x).split(E, 2)  # all three are of shape B x I X E        
        # splitting matrices into chunks aka head for performance
        q = q.view(B, I, self.head_count,  E//self.head_count).permute(0, 2, 1, 3)
        k = k.view(B, I, self.head_count,  E//self.head_count).permute(0, 2, 1, 3)
        v = v.view(B, I, self.head_count,  E//self.head_count).permute(0, 2, 1, 3)

        # now this is point where splitting into heads changes the results
        # because multiplying QxK isn't the same multiplying chunks of Q and K and bringing them back together
        x = q @ k.transpose(-2, -1)  # batch_size x input_size x input_size
        x = x*x.shape[-1]**-0.5  # scale result 1/sqrt(d_k) in the paper
        x = x.masked_fill(self.tril[:, :, :I, :I] == 0, float('-inf'))
        x = F.softmax(x, dim=-1)
        x = self.dropout(x)
        # batch_size x input_size x embd_size
        x = x @ v
        x = x.permute(0, 2, 1, 3).reshape(B, I, E)
        return x
