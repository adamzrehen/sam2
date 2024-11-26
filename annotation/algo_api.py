import torch
import gc
import os
import shutil
import cv2
import numpy as np
import ffmpeg
import gradio as gr
from utils import clear_folder, mask2bbox, draw_rect, draw_markers, show_mask
try:
    from sam2.build_sam import build_sam2
except ImportError:
    raise ImportError("Please install the SAM2 package. "
                      "See https://pypi.org/project/sam2/ and dpwnload MedSem weights")

from sam2.utils.transforms import SAM2Transforms
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor


class AlgoAPI:
    def __init__(self):
        pass

    @staticmethod
    def clean(seg_tracker):
        if seg_tracker is not None:
            predictor, inference_state, image_predictor = seg_tracker
            predictor.reset_state(inference_state)
            del predictor
            del inference_state
            del image_predictor
            del seg_tracker
            gc.collect()
            torch.cuda.empty_cache()
        return None, ({}, {}), None, None, 0, None, None, None, 0

    @staticmethod
    def verify_upload(video_path):
        if video_path is None:
            print("No video file uploaded")
            return None

        print(f"Uploaded video path: {video_path}")

        try:
            # Check if file exists and is not empty
            if not os.path.exists(video_path):
                print(f"File does not exist: {video_path}")
                return None

            file_size = os.path.getsize(video_path)
            print(f"File size in tmp location: {file_size / (1024 * 1024):.2f} MB")

            if file_size == 0:
                print("Warning: File exists but is empty")

                # Check tmp directory space
                tmp_dir = os.path.dirname(video_path)
                total, used, free = shutil.disk_usage(tmp_dir)
                print(f"Tmp directory space - Total: {total / (1024 ** 3):.1f}GB, "
                      f"Used: {used / (1024 ** 3):.1f}GB, "
                      f"Free: {free / (1024 ** 3):.1f}GB")

                return None

            # Try to read first few bytes
            try:
                with open(video_path, 'rb') as f:
                    first_bytes = f.read(1024)
                    if not first_bytes:
                        print("Warning: File is not readable")
                        return None
            except IOError as e:
                print(f"Error reading file: {e}")
                return None

            return video_path

        except Exception as e:
            print(f"Error accessing file: {e}")
            return None

    def get_meta_from_video(self, seg_tracker, input_video, scale_slider, checkpoint):
        # Verify upload first
        verified_path = self.verify_upload(input_video)
        if verified_path is None:
            print("Video upload verification failed")
            return None, ({}, {}), None, None, 0, None, None, None, 0

        output_dir = 'output_frames'
        output_masks_dir = 'output_masks'
        output_combined_dir = 'output_combined'
        clear_folder(output_dir)
        clear_folder(output_masks_dir)
        clear_folder(output_combined_dir)
        if input_video is None:
            return None, ({}, {}), None, None, 0, None, None, None, 0
        cap = cv2.VideoCapture(input_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert total_frames > 0, "No frames found in the input video"
        cap.release()
        output_frames = int(total_frames * scale_slider)
        frame_interval = max(1, total_frames // output_frames)
        ffmpeg.input(input_video, hwaccel='cuda').output(
            os.path.join(output_dir, '%07d.jpg'),
            q=2,
            start_number=0,
            vf=rf'select=not(mod(n\,{frame_interval}))',
            vsync='vfr'
        ).run()

        first_frame_path = os.path.join(output_dir, '0000000.jpg')
        first_frame = cv2.imread(first_frame_path)
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

        if seg_tracker is not None:
            del seg_tracker
            seg_tracker = None
            gc.collect()
            torch.cuda.empty_cache()
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if checkpoint == "tiny":
            sam2_checkpoint = "checkpoints/sam2_hiera_tiny.pt"
            model_cfg = "sam2_hiera_t.yaml"
        elif checkpoint == "small":
            sam2_checkpoint = "checkpoints/sam2_hiera_small.pt"
            model_cfg = "sam2_hiera_s.yaml"
        elif checkpoint == "base-plus":
            sam2_checkpoint = "checkpoints/sam2_hiera_base_plus.pt"
            model_cfg = "sam2_hiera_b+.yaml"
        elif checkpoint == "large":
            sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
            model_cfg = "sam2_hiera_l.yaml"
        else:
            raise ValueError(f"Invalid checkpoint: {checkpoint}")

        assert os.path.exists(sam2_checkpoint), (f"Please download MedSem Checkpoint from:\n"
                                                 f"https://github.com/SuperMedIntel/Medical-SAM2/blob/main/checkpoints/download_ckpts.sh \n"
                                                 f"and place in:\n"
                                                 f"checkpoint/")
        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device="cuda")
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

        image_predictor = SAM2ImagePredictor(sam2_model)
        inference_state = predictor.init_state(video_path=output_dir)
        predictor.reset_state(inference_state)
        return (predictor, inference_state, image_predictor), (
            {}, {}), first_frame_rgb, first_frame_rgb, 0, None, None, None, 0

    def sam_stroke(self, seg_tracker, drawing_board, last_draw, frame_num, ann_obj_id):
        predictor, inference_state, image_predictor = seg_tracker
        image_path = f'output_frames/{frame_num:07d}.jpg'
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        display_image = drawing_board["image"]
        image_predictor.set_image(image)
        input_mask = drawing_board["mask"]
        input_mask[input_mask != 0] = 255
        if last_draw is not None:
            diff_mask = cv2.absdiff(input_mask, last_draw)
            input_mask = diff_mask
        bbox, hasMask = mask2bbox(input_mask[:, :, 0])
        if not hasMask:
            return seg_tracker, display_image, display_image
        masks, scores, logits = image_predictor.predict(point_coords=None, point_labels=None, box=bbox[None, :],
                                                        multimask_output=False, )
        mask = masks > 0.0
        masked_frame = show_mask(mask, display_image, ann_obj_id)
        masked_with_rect = draw_rect(masked_frame, bbox, ann_obj_id)
        frame_idx, object_ids, masks = predictor.add_new_mask(inference_state, frame_idx=frame_num, obj_id=ann_obj_id,
                                                              mask=mask[0])
        last_draw = drawing_board["mask"]
        return seg_tracker, masked_with_rect, masked_with_rect, last_draw

    def sam_click(self, seg_tracker, frame_num, point_mode, click_stack, ann_obj_id, evt: gr.SelectData):
        points_dict, labels_dict = click_stack
        predictor, inference_state, image_predictor = seg_tracker
        ann_frame_idx = frame_num  # the frame index we interact with
        print(f'ann_frame_idx: {ann_frame_idx}')
        point = np.array([[evt.index[0], evt.index[1]]], dtype=np.float32)
        if point_mode == "Positive":
            label = np.array([1], np.int32)
        else:
            label = np.array([0], np.int32)

        if ann_frame_idx not in points_dict:
            points_dict[ann_frame_idx] = {}
        if ann_frame_idx not in labels_dict:
            labels_dict[ann_frame_idx] = {}

        if ann_obj_id not in points_dict[ann_frame_idx]:
            points_dict[ann_frame_idx][ann_obj_id] = np.empty((0, 2), dtype=np.float32)
        if ann_obj_id not in labels_dict[ann_frame_idx]:
            labels_dict[ann_frame_idx][ann_obj_id] = np.empty((0,), dtype=np.int32)

        points_dict[ann_frame_idx][ann_obj_id] = np.append(points_dict[ann_frame_idx][ann_obj_id], point, axis=0)
        labels_dict[ann_frame_idx][ann_obj_id] = np.append(labels_dict[ann_frame_idx][ann_obj_id], label, axis=0)

        click_stack = (points_dict, labels_dict)

        frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points_dict[ann_frame_idx][ann_obj_id],
            labels=labels_dict[ann_frame_idx][ann_obj_id],
        )

        image_path = f'output_frames/{ann_frame_idx:07d}.jpg'
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masked_frame = image.copy()
        for i, obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()
            masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)
        masked_frame_with_markers = draw_markers(masked_frame, points_dict[ann_frame_idx], labels_dict[ann_frame_idx])

        return seg_tracker, masked_frame_with_markers, masked_frame_with_markers, click_stack
