import torch
from sam2.build_sam import build_sam2_video_predictor
import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam2.utils.amg import build_all_layer_point_grids
import os


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
    32,
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
        frame_idx, object_ids, masks = predictor.add_new_points(
            state,
            frame_idx=0,
            obj_id=i + 1,
            points=points,
            labels=labels,
        )
        if i == 2:
            break
    print(len(masks))
    raise
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    # render the segmentation results every few frames
    frame_num = 0
    for out_frame_idx in range(0, len(video_segments.keys())):
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            plt.imshow(out_mask[0])
            plt.savefig(f"results/{str(frame_num).zfill(4)}.png")
            frame_num += 1
