import torch
from sam2.build_sam import build_sam2_video_predictor, build_sam2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam2.utils.amg import build_all_layer_point_grids
import os
import math
import shutil
from data import HCIOldDataset
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

BATCH_SIZE = 10
MODEL_CHECKPOINT = "checkpoints/sam2_hiera_tiny.pt"
MODEL_CONFIG = "sam2_hiera_t.yaml"


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


def LF_lawnmower(LF):
    result_LF = []
    rows, cols, u, v = LF.shape
    for i in range(rows):
        if i % 2 == 0:
            for j in range(cols):
                result_LF.append(LF[i, j])
        else:
            for j in range(cols - 1, -1, -1):
                result_LF.append(LF[i, j])
    result_LF = torch.stack(result_LF).reshape(rows, cols, u, v)
    return result_LF


def get_image_predictor():
    predictor = SAM2AutomaticMaskGenerator(
        build_sam2(
            MODEL_CONFIG,
            MODEL_CHECKPOINT,
            device="cuda",
            apply_postprocessing=False,
        )
    )
    return predictor


def get_video_predictor():
    predictor = build_sam2_video_predictor(MODEL_CONFIG, MODEL_CHECKPOINT)
    return predictor


def get_subview_masks(image_predictor, subview):
    result = image_predictor.generate(subview)
    result = torch.stack([torch.tensor(x["segmentation"]).cuda() for x in result])
    return result


def batchify_masks(masks):
    return torch.tensor_split(masks, math.ceil(masks.shape[0] / BATCH_SIZE))


def propagate_masks(masks_batchified, video_predictor):
    for batch_i, batch in enumerate(masks_batchified):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = video_predictor.init_state("LF")
            for i, mask in enumerate(batch):
                frame_idx, object_ids, masks = video_predictor.add_new_mask(
                    state,
                    frame_idx=0,
                    obj_id=i + 1,
                    mask=mask,
                )
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in video_predictor.propagate_in_video(state):
                out_mask_logits = out_mask_logits[:, 0, :, :] > 0.0
                masks_result = out_mask_logits.to_sparse()
                torch.save(
                    masks_result,
                    f"results/{str(out_frame_idx).zfill(4)}_{str(batch_i).zfill(4)}.pt",
                )


def save_results(LF_original, result_filename="LF.pt", batch_size=BATCH_SIZE):
    s, t, u, v, _ = LF_original.shape
    n_imgs = s * t
    LF_masks = []
    for img_i in range(n_imgs):
        batch_masks = []
        for batch_i in range(batch_size):
            filename = f"results/{str(img_i).zfill(4)}_{str(batch_i).zfill(4)}.pt"
            if not os.path.exists(filename):
                continue
            masks = torch.load(filename)
            batch_masks.append(masks)
            os.remove(filename)
        batch_masks = torch.cat(batch_masks, dim=0).to_dense()
        areas = batch_masks.sum(dim=(1, 2))
        masks_result = torch.zeros_like(batch_masks[0]).long()
        for i, _ in enumerate(torch.argsort(areas, descending=True)):
            masks_result[batch_masks[i]] += i
            masks_result = torch.clip(masks_result, 0, i).long()
        del batch_masks
        del areas
        LF_masks.append(masks_result)
        del masks_result
    LF_masks = torch.stack(LF_masks).cuda().reshape(s, t, u, v)
    LF_masks = LF_lawnmower(LF_masks)
    torch.save(LF_masks, f"results/{result_filename}")
    del LF_masks


def main():
    img_predictor = get_image_predictor()
    video_predictor = get_video_predictor()
    dataset = HCIOldDataset("HCI_dataset_old")
    for i, item in enumerate(dataset):
        LF, _, _ = item
        subview = LF[0][0]
        masks = get_subview_masks(img_predictor, subview)
        save_LF_lawnmower(LF)
        masks_batchified = batchify_masks(masks)
        propagate_masks(masks_batchified, video_predictor)
        save_results(
            LF,
            f"{str(i).zfill(4)}_result.pth",
        )
        shutil.rmtree("LF")


if __name__ == "__main__":
    main()
