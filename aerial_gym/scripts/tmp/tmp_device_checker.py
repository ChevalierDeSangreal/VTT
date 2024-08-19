import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
import torch



device = torch.device('cuda:1')
torch.cuda.set_device(device)
print(torch.cuda.current_device())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(1))
tmp = torch.tensor([0, 0, 0], device=device)
print(tmp)

