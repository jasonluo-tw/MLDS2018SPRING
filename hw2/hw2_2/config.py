import torch
USE_CUDA = torch.cuda.is_available()

MAX_LENGTH = 8
teacher_forcing_ratio = 0.7 
save_dir = './save'
