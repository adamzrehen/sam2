import torch
import gc
import os
import cv2
import numpy as np
import gradio as gr
import pickle
from utils import mask2bbox, draw_rect, draw_markers, show_mask
try:
    from sam2.build_sam import build_sam2
except ImportError:
    raise ImportError("Please install the SAM2 package. "
                      "See https://pypi.org/project/sam2/ and dpwnload MedSem weights")

from sam2.utils.transforms import SAM2Transforms
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor


class AlgoAPI:
    def __init__(self, base_dir, config):
        self.base_dir = base_dir
        self.config = config
        self.segment_id = None
        self.video = None
        self.segment_masks = {}
        self.seg_tracker = None

    def update_segment_id(self, segment_id):
        self.segment_id = segment_id

    def update_video(self, video):
        self.video = video
        self.segment_id = 0

    def clean(self):
        if self.seg_tracker is not None:
            predictor, inference_state, image_predictor = self.seg_tracker
            predictor.reset_state(inference_state)
            del predictor, inference_state, image_predictor, self.seg_tracker
            gc.collect()
            torch.cuda.empty_cache()
        return None, ({}, {}), None, None, 0, None, None, None, 0

    def initialize_sam(self, checkpoint, output_paths):
        if not torch.cuda.is_available():
            return None

        if self.seg_tracker is not None:
            del self.seg_tracker
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
        inference_state = predictor.init_state(video_path=output_paths['frames_dir'])

        self.seg_tracker = (predictor, inference_state, image_predictor)
        predictor.reset_state(inference_state)
        num_frames = inference_state['images'].shape[0]

        # Load existing point_inputs_per_obj if available
        segment_name = os.path.basename(self.video['segments'][self.segment_id]['path'].split('.mp4')[0])
        inference_state_path = os.path.join(self.base_dir,
                                            os.path.dirname(self.video['segments'][self.segment_id]['path']),
                                            f'click_stack_{segment_name}.pkl')
        click_stack = ({}, {})
        if os.path.exists(inference_state_path):
            click_stack = self.load_click_stack()
            self.add_points(click_stack)

        return click_stack, num_frames

    def add_points(self, click_stack):
        points_dict, labels_dict = click_stack
        predictor, inference_state, image_predictor = self.seg_tracker
        for frame_num, vals in points_dict.items():
                for obj_id in vals.keys():
                    frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points(
                        inference_state=inference_state,
                        frame_idx=frame_num,
                        obj_id=obj_id,
                        points=points_dict[frame_num][obj_id],
                        labels=labels_dict[frame_num][obj_id],
                    )
                    self.segment_masks[frame_num] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

    def propagate(self):
        predictor, inference_state, image_predictor = self.seg_tracker
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            self.segment_masks[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    @staticmethod
    def pack_to_tuple(nested_dict):
        coords_dict, labels_dict = {}, {}
        for key, subdict in nested_dict.items():
            coords_dict[key] = {}
            labels_dict[key] = {}
            for subkey, data in subdict.items():
                coords_dict[key][subkey] = data['point_coords'][0].cpu().numpy()
                labels_dict[key][subkey] = data['point_labels'][0].cpu().numpy()
        return coords_dict, labels_dict

    def sam_stroke(self, drawing_board, last_draw, frame_num, ann_obj_id):
        predictor, inference_state, image_predictor = self.seg_tracker
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
            return display_image, display_image
        masks, scores, logits = image_predictor.predict(point_coords=None, point_labels=None, box=bbox[None, :],
                                                        multimask_output=False, )
        mask = masks > 0.0
        masked_frame = show_mask(mask, display_image, ann_obj_id)
        masked_with_rect = draw_rect(masked_frame, bbox, ann_obj_id)
        frame_idx, object_ids, masks = predictor.add_new_mask(inference_state, frame_idx=frame_num, obj_id=ann_obj_id,
                                                              mask=mask[0])
        last_draw = drawing_board["mask"]
        return masked_with_rect, masked_with_rect, last_draw

    def sam_click(self, frame_num, point_mode, click_stack, ann_obj_id, evt: gr.SelectData):
        points_dict, labels_dict = click_stack
        predictor, inference_state, image_predictor = self.seg_tracker
        ann_frame_idx = frame_num  # the frame index we interact with
        point = np.array([[evt.index[0], evt.index[1]]], dtype=np.float32)
        label = np.array([1], np.int32) if point_mode == "Positive" else np.array([0], np.int32)

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
            clear_old_points=True,
        )

        image_path = os.path.join(self.video['video_metadata']['output_dir'], f'output_frames/{ann_frame_idx:07d}.jpg')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masked_frame = image.copy()
        for i, obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()
            masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)
        masked_frame_with_markers = draw_markers(masked_frame, points_dict[ann_frame_idx], labels_dict[ann_frame_idx])

        self.save_click_stack(click_stack)
        return masked_frame_with_markers, masked_frame_with_markers, click_stack

    def save_click_stack(self, click_stack):
        segment_name = os.path.basename(self.video['segments'][self.segment_id]['path'].split('.mp4')[0])
        output_path = os.path.join(self.base_dir, 'data/sam2', self.video['video_metadata']['video_name'],
                                   f'segments/click_stack_{segment_name}' + '.pkl')
        with open(output_path, 'wb') as file:
            pickle.dump(click_stack, file)

    def load_click_stack(self):
        segment_name = os.path.basename(self.video['segments'][self.segment_id]['path'].split('.mp4')[0])
        input_path = os.path.join(self.base_dir, 'data/sam2', self.video['video_metadata']['video_name'],
                                  f'segments/click_stack_{segment_name}' + '.pkl')
        with open(input_path, 'rb') as file:
            click_stack = pickle.load(file)
        return click_stack
