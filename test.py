import torch
from sam2.build_sam import build_sam2_video_predictor
import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam2.utils.amg import build_all_layer_point_grids
import os
import imgviz


checkpoint = "/home/cedaradmin/repos/segment-anything-2/checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)


# scene = h5py.File(
#     "/home/cedaradmin/repos/LF_SAM_segmentation/HCI_dataset_old/papillon/lf.h5",
#     "r",
# )
# LF = np.array(scene["LF"])
# print(LF.shape)

# def snake_pattern(matrix):
#     rows, cols, _, _, _ = matrix.shape
#     frame_n = 0
#     for i in range(rows):
#         if i % 2 == 0:
#             for j in range(cols):
#                 Image.fromarray(matrix[i, j]).save(f"LF/{str(frame_n).zfill(4)}.jpeg")
#                 frame_n += 1
#         else:
#             for j in range(cols - 1, -1, -1):
#                 Image.fromarray(matrix[i, j]).save(f"LF/{str(frame_n).zfill(4)}.jpeg")
#                 frame_n += 1


# snake_pattern(LF)

# print(LF.shape)

points = build_all_layer_point_grids(
    4,
    0,
    1,
)
points = np.stack(points)[0]
points = points * np.array([768.0, 768.0])
labels = np.ones((points.shape[0])).astype(np.int32)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state("/home/cedaradmin/repos/segment-anything-2/LF")

    # add new prompts and instantly get the output on the same frame
    for i, (point, label) in enumerate(zip(points, labels)):
        # np.random.shuffle(points)
        frame_idx, object_ids, masks = predictor.add_new_points(
            state,
            frame_idx=0,
            obj_id=i + 1,
            points=point[None],
            labels=label[None],
        )
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        state
    ):
        masks = torch.stack(
            [(mask > 0.0).long()[0] * (i + 1) for i, mask in enumerate(out_mask_logits)]
        ).cuda()
        areas = masks.sum(dim=(1, 2))
        masks_result = torch.zeros_like(masks[0])
        for i, mask_ind in enumerate(torch.argsort(areas, descending=True)):
            masks_result += masks[mask_ind] * i
            masks_result = torch.clip(masks_result, 0, i).long()
        torch.save(masks_result, f"results/{str(out_frame_idx).zfill(4)}.pt")
        # vis = imgviz.label2rgb(
        #     label=masks_result.cpu().numpy(),
        #     # image=np.array(Image.open(f"LF/{str(out_frame_idx).zfill(4)}.jpeg")),
        #     colormap=imgviz.label_colormap(masks_result.max() + 1),
        # )
        # plt.savefig(f"results/{str(out_frame_idx).zfill(4)}.png")
