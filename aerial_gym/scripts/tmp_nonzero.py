import torch
reset_buf = torch.tensor([0, 0, 0])
reset_idx = torch.nonzero(reset_buf).squeeze(-1)
print(reset_idx)
if not len(reset_idx):
    print("???????")