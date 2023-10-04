# -*-coding:utf-8-*-
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F

a = np.load('ImageNet_SIG.npy').astype(np.uint8)
print(a)
# a = a.reshape((32, 32, 1))
# b = np.repeat(a, 3, 2)
c = Image.fromarray(a)
c.show()
#
# with open('signal_cifar10_mask.png', "rb") as f:
#     trigger_ptn = Image.open(f).convert("RGB")
#     a = np.array(trigger_ptn)
#     print(a.shape)

# def create_grids(img, k=4, s=0.5, grid_rescale=1):
#     height = img.shape[0]
#     ins = torch.rand(1, 2, k, k) * 2 - 1
#     ins = ins / torch.mean(torch.abs(ins))
#     noise_grid = (
#         F.upsample(ins, size=height, mode="bicubic", align_corners=True)
#             .permute(0, 2, 3, 1)
#     )
#     array1d = torch.linspace(-1, 1, steps=height)  # (32)
#     x, y = torch.meshgrid(array1d, array1d)
#
#     identity_grid = torch.stack((y, x), 2)[None, ...] # (1, 32, 32, 2)
#     grid_temps = (identity_grid + s * noise_grid / height) * grid_rescale
#     bd_temps = torch.clamp(grid_temps, -1, 1)
#
#     # ins = torch.rand(num_cross, height, height, 2) * 2 - 1
#     # grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + ins / height
#     # noisy_temps = torch.clamp(grid_temps2, -1, 1)
#
#     return bd_temps
#
# bd_grids = create_grids(np.ones((224, 224, 3)), k=224, s=1)
# torch.save(bd_grids, 'ImageNet_WaNet_bd_grid.pt')

