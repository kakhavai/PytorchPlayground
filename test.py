import torch
x = torch.rand(5, 3)
print('VERIFY TENSOR OUTPUT: \n' + str(x) + '\n')
print('CUDA AVAILABILITY: ' + str(torch.cuda.is_available()))