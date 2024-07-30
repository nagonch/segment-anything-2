import torch
from PIL import Image
import numpy as np
import imgviz
import matplotlib.pyplot as plt
import torch

for i in range(81):
    mask = torch.load(f"results/{str(i).zfill(4)}.pt").cpu().numpy()
    image = np.array(Image.open(f"LF/{str(i).zfill(4)}.jpeg"))
    vis = imgviz.label2rgb(
        label=mask,
        image=image,
        colormap=imgviz.label_colormap(mask.max() + 1),
    )
    plt.imshow(vis)
    plt.savefig(f"results/{str(i).zfill(4)}.png")
