import torch
from sam2.build_sam import build_sam2_video_predictor, build_sam2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam2.utils.amg import build_all_layer_point_grids
import os
import shutil
from data import HCIOldDataset
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

BATCH_SIZE = 16
MODEL_CHECKPOINT = "checkpoints/sam2_hiera_tiny.pt"
MODEL_CONFIG = "sam2_hiera_t.yaml"


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


def inference(predictor, LF, batch_size=BATCH_SIZE):
    s, t, u, v, c = LF.shape
    points = build_all_layer_point_grids(
        8,
        0,
        1,
    )
    points = np.stack(points)[0]
    points = points * np.array([float(u), float(v)])
    labels = np.ones((points.shape[0])).astype(np.int32)
    n_batches = np.ceil(points.shape[0] / batch_size)
    points = np.array_split(points, n_batches)
    labels = np.array_split(labels, n_batches)
    for batch_i, (points_batch, labels_batch) in enumerate(zip(points, labels)):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state("LF")
            for i, (point, label) in enumerate(zip(points_batch, labels_batch)):
                frame_idx, object_ids, masks = predictor.add_new_points(
                    state,
                    frame_idx=0,
                    obj_id=i + 1,
                    points=point[None],
                    labels=label[None],
                )
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in predictor.propagate_in_video(state):
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
        max_segment_num = 0
        for batch_i in range(batch_size):
            filename = f"results/{str(img_i).zfill(4)}_{str(batch_i).zfill(4)}.pt"
            if not os.path.exists(filename):
                continue
            mask = torch.load(filename)
            mask[mask != 0] += max_segment_num
            max_segment_num = mask.max()
            batch_masks.append(mask)
            os.remove(filename)
        masks_i = torch.stack(batch_masks)
        masks_i = torch.sum(masks_i, dim=0)
        LF_masks.append(masks_i)
    LF_masks = torch.stack(LF_masks).cuda().reshape(s, t, u, v)
    LF_masks = LF_lawnmower(LF_masks)
    torch.save(LF_masks, f"results/{result_filename}")
    del LF_masks


def main():
    dataset = HCIOldDataset("HCI_dataset_old")
    for i, item in enumerate(dataset):
        LF, _, _ = item
        save_LF_lawnmower(LF)
        predictor = get_video_predictor()
        inference(predictor, LF)
        del predictor
        torch.cuda.empty_cache()
        save_results(
            LF,
            f"{str(i).zfill(4)}_result.pth",
        )
        shutil.rmtree("LF")


if __name__ == "__main__":
    img_predictor = get_image_predictor()
    # video_predictor = get_video_predictor()
    dataset = HCIOldDataset("HCI_dataset_old")
    LF, _, _ = dataset[0]
    subview_1 = LF[0][0]
    print(get_subview_masks(img_predictor, subview_1).shape)
