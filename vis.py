import h5py
import numpy as np
import imgviz
from plenpy.lightfields import LightField
import torch


# Load LF
def get_LF():
    scene = h5py.File(
        "/home/cedaradmin/repos/LF_SAM_segmentation/HCI_dataset_old/papillon/lf.h5",
        "r",
    )
    LF = np.array(scene["LF"])
    return LF


def visualize_segmentation_mask(
    segments,
    LF=None,
    just_return=False,
):
    s, t, u, v = segments.shape
    segments = np.transpose(segments, (0, 2, 1, 3)).reshape(s * u, t * v)
    if LF is not None:
        LF = np.transpose(LF, (0, 2, 1, 3, 4)).reshape(s * u, t * v, 3)
    vis = np.transpose(
        imgviz.label2rgb(
            label=segments,
            image=LF,
            colormap=imgviz.label_colormap(segments.max() + 1),
        ).reshape(s, u, t, v, 3),
        (0, 2, 1, 3, 4),
    )
    if not just_return:
        segments = LightField(vis)
        segments.show()
    return vis


if __name__ == "__main__":
    LF = get_LF()
    masks = torch.load("results/LF.pt").cpu().numpy()
    visualize_segmentation_mask(masks, LF)
