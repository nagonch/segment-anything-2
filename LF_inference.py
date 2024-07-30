import torch
from sam2.build_sam import build_sam2_video_predictor
import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam2.utils.amg import build_all_layer_point_grids
import os
import imgviz
import shutil


# Load LF
def get_LF():
    scene = h5py.File(
        "/home/cedaradmin/repos/LF_SAM_segmentation/HCI_dataset_old/papillon/lf.h5",
        "r",
    )
    LF = np.array(scene["LF"])
    return LF


# Save LF temp save LF to folder
def save_LF_lawnmower(LF):
    os.makedirs("LF", exist_ok=True)
    rows, cols, _, _, _ = LF.shape
    frame_n = 0
    for i in range(rows):
        if i % 2 == 0:
            for j in range(cols):
                Image.fromarray(LF[i, j]).save(f"LF/{str(frame_n).zfill(4)}.jpeg")
                frame_n += 1
        else:
            for j in range(cols - 1, -1, -1):
                Image.fromarray(LF[i, j]).save(f"LF/{str(frame_n).zfill(4)}.jpeg")
                frame_n += 1


# Load predictor
def get_predictor():
    checkpoint = (
        "/home/cedaradmin/repos/segment-anything-2/checkpoints/sam2_hiera_tiny.pt"
    )
    model_cfg = "sam2_hiera_t.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)
    return predictor


def inference(predictor, LF):
    s, t, u, v, c = LF.shape
    points = build_all_layer_point_grids(
        4,
        0,
        1,
    )
    points = np.stack(points)[0]
    points = points * np.array([float(u), float(v)])
    labels = np.ones((points.shape[0])).astype(np.int32)
    result_mask_LF = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state("/home/cedaradmin/repos/segment-anything-2/LF")
        for i, (point, label) in enumerate(zip(points, labels)):
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
                [
                    (mask > 0.0).long()[0] * (i + 1)
                    for i, mask in enumerate(out_mask_logits)
                ]
            ).cuda()
            areas = masks.sum(dim=(1, 2))
            masks_result = torch.zeros_like(masks[0])
            for i, mask_ind in enumerate(torch.argsort(areas, descending=True)):
                masks_result += masks[mask_ind] * i
                masks_result = torch.clip(masks_result, 0, i).long()
            torch.save(masks_result, f"results/{str(out_frame_idx).zfill(4)}.pt")


# Clean up LF folder
if __name__ == "__main__":
    LF = get_LF()
    save_LF_lawnmower(LF)
    predictor = get_predictor()
    inference(predictor, LF)
    shutil.rmtree("LF")
