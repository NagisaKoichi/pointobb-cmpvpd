import torch
A = torch.tensor([[[float('nan'), 0.0], [0.0, 0.0]]])
try:
    torch.linalg.eigh(A)
except Exception as e:
    print("NAN:", type(e), e)
