import torch
from nanoshakes import NanoShakes
from lptimer import LP_Timer as Timer

# load data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
char2ind = {ch: i for i, ch in enumerate(vocab)}
ind2char = {i: ch for i, ch in enumerate(vocab)}
def encode(x): return [char2ind[c] for c in x]
def decode(x): return ''.join(ind2char[i] for i in x)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

vocab_size = 65
input_size = 256
embd_size = 384
head_count = 6
layer_count = 6
dropout = 0.2

print(f'Device: {device}')

trained_model = NanoShakes(vocab_size=vocab_size, input_size=input_size, embd_size=embd_size,
                           layer_count=layer_count, head_count=head_count, dropout=dropout, device=device)

trained_model.load_state_dict(torch.load(
    'trained.pth', map_location=torch.device(device)))
trained_model.to(torch.device(device))
start = torch.zeros((1, 1), dtype=torch.long, device=device)
timer = Timer().start()
print(decode(trained_model.generate(200)))
print(timer.stop())
