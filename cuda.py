import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)