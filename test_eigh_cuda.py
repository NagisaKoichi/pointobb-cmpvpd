import torch
for A in [
    torch.tensor([[[0.0, 0.0], [0.0, 0.0]]]),
    torch.tensor([[[float('nan'), 0.0], [0.0, 0.0]]]),
    torch.tensor([[[float('inf'), 0.0], [0.0, 0.0]]]),
]:
    A = A.cuda()
    try:
        torch.linalg.eigh(A)
    except Exception as e:
        print("Error on", A, ":", e)
