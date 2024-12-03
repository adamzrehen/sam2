import os
import cv2
import math
import shutil
import gradio as gr
import ffmpeg
import numpy as np
import pickle

import pandas as pd

from utils import draw_markers, show_mask, zip_folder
from file_handler import (clear_folder, get_meta_from_video, load_click_stack, save_click_stack,
                          load_existing_video_metadata)
from algo_api import AlgoAPI


class Backend:
    def __init__(self, config, base_dir):
        self.base_dir = base_dir
        self.config = config
        self.algo_api = AlgoAPI(base_dir=self.base_dir, config=self.config)
        self.video = {}
        self.segmentation_state = True
        self.segment_id = 0
        self.ann_obj_id = 0
        self.frame_num = 0
        self.tissue_info = {}

    def move_slider(self, frame_num, click_stack):
        image_path = os.path.join(self.video['video_metadata']['output_dir'], 'output_frames')
        output_combined_dir = os.path.join(self.video['video_metadata']['output_dir'], 'output_combined')

        combined_frames = sorted(
            [os.path.join(output_combined_dir, img_name) for img_name in os.listdir(output_combined_dir)])
        if combined_frames:
            output_masked_frame_path = combined_frames
        else:
            original_frames = sorted([os.path.join(image_path, img_name) for img_name in os.listdir(image_path)])
            output_masked_frame_path = original_frames

        total_frames_num = len(output_masked_frame_path)
        if total_frames_num == 0:
            print("No output results found")
            return None, None
        else:
            self.frame_num = frame_num
            tissue_info = self.update_tissue_info()
            masked_frame = self.get_masked_frame(click_stack)
            return masked_frame, frame_num, *tissue_info

    def tracking_objects(self, click_stack):
        input_video = self.video['segments'][self.segment_id]['path']
        output_keys = ['frames_dir', 'masks_dir', 'combined_dir', 'video_path',
                       'zip_path']
        output_paths = {key: os.path.join(self.base_dir, self.video['video_metadata']['output_dir'], self.config[key])
                        for key in output_keys}

        for key in ['masks_dir', 'combined_dir']:
            clear_folder(output_paths[key])

        if os.path.exists(output_paths['video_path']):
            os.remove(output_paths['video_path'])
        if os.path.exists(output_paths['zip_path']):
            os.remove(output_paths['zip_path'])

        # Track and update segmentation masks
        self.algo_api.propagate()

        frame_files = sorted([f for f in os.listdir(output_paths['frames_dir']) if f.endswith('.jpg')])
        for frame_id, frame_file in enumerate(frame_files):
            frame_idx = int(os.path.splitext(frame_file)[0])
            masked_frame = self.get_masked_frame(click_stack, mask_dir=output_paths['masks_dir'])
            combined_output_path = os.path.join(output_paths['combined_dir'], f'{frame_idx:07d}.png')
            combined_image_bgr = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(combined_output_path, combined_image_bgr)
            if frame_idx == self.frame_num:
                final_masked_frame = masked_frame

        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        output_frames = len([name for name in os.listdir(output_paths['combined_dir']) if
                             os.path.isfile(os.path.join(output_paths['combined_dir'], name)) and name.endswith('.png')])
        out_fps = fps * output_frames / total_frames
        ffmpeg.input(os.path.join(output_paths['combined_dir'], '%07d.png'), framerate=out_fps).output(output_paths['video_path'],
                                                                                              vcodec='h264_nvenc',
                                                                                              pix_fmt='yuv420p').run()
        zip_folder(output_paths['masks_dir'], output_paths['zip_path'])
        return final_masked_frame, final_masked_frame

    def increment_ann_obj_id(self):
        self.ann_obj_id += 1

    @staticmethod
    def drawing_board_get_input_first_frame(input_first_frame):
        return input_first_frame

    def undo_last_point(self, click_stack):
        # Remove the last point and its corresponding label
        points_dict, labels_dict = click_stack
        if self.frame_num in points_dict and self.frame_num in labels_dict:
            for obj_id in points_dict[self.frame_num]:
                if len(points_dict[self.frame_num][obj_id]) > 0:
                    points_dict[self.frame_num][obj_id] = points_dict[self.frame_num][obj_id][:-1]
                    labels_dict[self.frame_num][obj_id] = labels_dict[self.frame_num][obj_id][:-1]

        # Redraw existing masks if any
        self.algo_api.add_points((points_dict, labels_dict))
        masked_frame = self.get_masked_frame(click_stack)
        return masked_frame, (points_dict, labels_dict)

    def toggle_segmentation(self, click_stack):
        self.segmentation_state = not self.segmentation_state
        masked_frame = self.get_masked_frame(click_stack)
        return masked_frame, masked_frame, click_stack

    @staticmethod
    def split_video(video_path, video_name, output_dir, progress_callback=None, max_size_mb=20, max_frames=100):
        # Splits a video into smaller segments based on size or number of frames.
        output_dir = os.path.join(output_dir, 'segments')
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Get video metadata using ffmpeg
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            if not video_stream:
                raise ValueError("No video stream found in the input file")

            total_frames = int(video_stream['nb_frames'])
            fps = eval(video_stream['avg_frame_rate'])
            duration = float(video_stream['duration'])
            bitrate = int(video_stream['bit_rate'])
            frame_duration = duration / total_frames

            # Calculate optimal segment size
            max_segment_duration = max_size_mb * 8 * 1024 * 1024 / bitrate
            max_segment_frames = min(max_frames, math.floor(max_segment_duration / frame_duration))

            segment_infos = []
            segment_start = 0
            segment_number = 0

            while segment_start < total_frames:
                segment_number += 1
                segment_end = min(segment_start + max_segment_frames, total_frames)

                # Calculate times
                start_time = segment_start / fps
                end_time = segment_end / fps

                segment_filename = os.path.join(output_dir, f"segment_{segment_number:04d}.mp4")

                # Split using ffmpeg
                ffmpeg.input(video_path, ss=start_time, t=(end_time - start_time)).output(
                    segment_filename, c='copy'
                ).run(overwrite_output=True)

                # Store segment info
                segment_info = {
                    'path': segment_filename,
                    'start_frame': segment_start,
                    'end_frame': segment_end,
                    'start_time': start_time,
                    'end_time': end_time,
                    'segment_number': segment_number
                }
                segment_infos.append(segment_info)

                if progress_callback:
                    progress = segment_number / math.ceil(total_frames / max_segment_frames) * 100
                    progress_callback(progress)

                segment_start = segment_end

            metadata = {
                "total_frames": total_frames,
                "fps": fps,
                "duration": duration,
                "bitrate": bitrate,
                "segments_created": segment_number,
                "video_name": video_name
            }

            return segment_infos, segment_number, metadata

        except Exception as e:
            print(f"Error splitting video: {e}")
            return None, 0, None

    def handle_large_video_upload(self, file_path):
        if file_path is None:
            return "No file uploaded", None, gr.Slider(), None
        if not os.path.exists(file_path.name):
            return f"ERROR File {file_path.name} does not exist", None, gr.Slider(), None

        file_size = os.path.getsize(file_path.name)
        if file_size < 0.2 * 1024 * 1024:
            return f"ERROR File {file_path.name} is too small", None, gr.Slider(), None

        try:
            # Get original filename
            video_metadata = {"video_name": os.path.splitext(os.path.basename(file_path.name))[0]}

            # set output directory
            video_metadata["output_dir"] = os.path.join(self.base_dir, 'data', 'sam2', video_metadata["video_name"])

            existing_video = load_existing_video_metadata(self, video_metadata)
            if not existing_video:
                os.makedirs(video_metadata["output_dir"])

                # Move file to output directory
                video_metadata["original_video_path"] = os.path.join(video_metadata["output_dir"],
                                                                     os.path.basename(file_path.name))
                shutil.move(file_path.name, video_metadata["original_video_path"])

                if not os.path.exists(video_metadata["original_video_path"]):
                    return (f'ERROR Failed to move file {file_path.name} to {video_metadata["original_video_path"]}',
                            None, gr.Slider(), None)

                yield f"Upload complete. Processing video...", None, gr.Slider(), video_metadata
                # Get initial video info
                try:
                    video_metadata["probe"] = ffmpeg.probe(video_metadata["original_video_path"])
                except Exception as e:
                    print(f"Error getting video info: {e}")

                video_metadata["video_info"] = next(s for s in video_metadata["probe"]['streams'] if s['codec_type'] == 'video')
                video_metadata["total_frames"] = int(video_metadata["video_info"]['nb_frames'])

                segment_infos, num_segments, metadata = self.split_video(
                    video_metadata["original_video_path"], video_metadata["video_name"], video_metadata["output_dir"],
                    progress_callback=lambda x: (f"Splitting video: {x:.1f}%", None, gr.Slider(), None)
                )
                self.video.update({
                    'video_metadata': video_metadata,
                    'metadata': metadata,
                    'paths': [segment_infos[i]['path'] for i in range(num_segments)],
                    'segments': segment_infos
                })

                # Save video object
                with open(os.path.join(self.base_dir, video_metadata["output_dir"],
                                       f'{video_metadata["video_name"]}_meta.pkl'), 'wb') as file:
                    pickle.dump(self.video, file)

            if self.video['segments']:
                msg = 'Video uploaded from database. Select a segment to begin.' if existing_video else \
                      'Processing complete. Select a segment to begin.'
                yield (
                    msg,
                    self.video['segments'][0]['path'],
                    gr.Slider(maximum=self.video['metadata']['segments_created'], value=1),
                )
            else:
                yield "Error: Failed to split video.", None, gr.Slider(), None

        except Exception as e:
            print(f"Error processing upload: {e}")
            yield f"Error during upload: {str(e)}", None, gr.Slider(), None

    def load_video_segment(self, index):
        self.segment_id = index
        return self.video['paths'][index - 1]

    def increment_video_index(self, current_index):
        """Increment video index while staying within bounds"""
        max_index = len(self.video['paths']) if self.video['paths'] else 1
        return min(max_index, current_index + 1)

    @staticmethod
    def decrement_video_index(current_index):
        """Decrement video index while staying within bounds"""
        return max(1, current_index - 1)

    def update_patient_info(self, *values):
        keys = ['Tagging Status', 'Camera', 'Annotator', 'Patient Code', 'Gender', 'Age', 'Height', 'Width', 'Thickness',
                'Tumor Height', 'Tumor Width', 'Depth', 'Grade', 'Stage', 'LVI', 'PNI', 'Deep Superficial',
                'Tissue Layer Invasion']
        patient_info = {}
        for key, val in zip(keys, values):
            patient_info[key] = [val]
        if 'video_metadata' in self.video:
            pd.DataFrame(patient_info).T.to_csv(os.path.join(self.video['video_metadata']['output_dir'],
                                                             'patient_data.csv'), index=True)

    def update_tissue_info(self, *values):
        keys = ['Filter', 'Perspective', 'Tumor Type', 'Indigo Carmine', 'Good Image', 'Indication', 'Site', 'Approved',
                'Health Status', 'Comment']
        if not len(self.tissue_info):
            for key, val in zip(keys, values):
                self.tissue_info[key] = [val]
            self.tissue_info['Segment ID'] = [self.segment_id]
            self.tissue_info['Frame Num'] = [self.frame_num]
            self.tissue_info = pd.DataFrame(self.tissue_info)
        else:
            filtered_rows = self.tissue_info[(self.tissue_info['Segment ID'] == self.segment_id) &
                                             (self.tissue_info['Frame Num'] == self.frame_num)]
            if len(filtered_rows):
                if len(values):
                    condition = (self.tissue_info['Segment ID'] == self.segment_id) & \
                                (self.tissue_info['Frame Num'] == self.frame_num)
                    for key, val in zip(keys, values):
                        self.tissue_info.loc[condition, key] = val
                    self.save_tissue_info()
                    return
            else:
                # Copy the last frame data to the new frame (sticky)
                filtered_rows = self.tissue_info[(self.tissue_info['Segment ID'] == self.segment_id) &
                                                 (self.tissue_info['Frame Num'] == self.frame_num - 1)]
                filtered_rows['Frame Num'] = self.frame_num
                self.tissue_info = pd.concat([self.tissue_info, filtered_rows], ignore_index=True)

        filtered_rows = self.tissue_info[(self.tissue_info['Segment ID'] == self.segment_id) &
                                         (self.tissue_info['Frame Num'] == self.frame_num)]
        filtered_rows = filtered_rows.drop(['Segment ID', 'Frame Num'], axis=1)
        self.save_tissue_info()
        return list(filtered_rows.values[0])

    def save_tissue_info(self):
        if 'video_metadata' in self.video:
            pd.DataFrame(self.tissue_info).T.to_csv(os.path.join(self.video['video_metadata']['output_dir'],
                                                                 'tissue_data.csv'), index=True)


    def sam_stroke(self, drawing_board, last_draw, frame_num):
        return self.algo_api.sam_stroke(drawing_board, last_draw, frame_num)

    def preprocess_video(self, scale_slider, checkpoint, *tissue_info):
        input_video = self.video['segments'][self.segment_id]['path']
        output_paths, first_frame_rgb = get_meta_from_video(self, input_video, scale_slider)
        click_stack = load_click_stack(self)
        tissue_info = self.update_tissue_info(*tissue_info)
        click_stack, num_frames = self.algo_api.initialize_sam(checkpoint, output_paths, click_stack)
        masked_frame = self.get_masked_frame(click_stack)
        return click_stack, masked_frame, gr.Slider(maximum=num_frames - 1, value=0), *tissue_info

    def sam_click(self, frame_num, point_mode, click_stack, evt: gr.SelectData):
        points_dict, labels_dict = click_stack
        ann_frame_idx = frame_num  # the frame index we interact with
        point = np.array([[evt.index[0], evt.index[1]]], dtype=np.float32)
        label = np.array([1], np.int32) if point_mode == "Positive" else np.array([0], np.int32)

        # Initialize nested dictionaries and arrays using setdefault
        points_dict.setdefault(ann_frame_idx, {}).setdefault(self.ann_obj_id, np.empty((0, 2), dtype=np.float32))
        labels_dict.setdefault(ann_frame_idx, {}).setdefault(self.ann_obj_id, np.empty((0,), dtype=np.int32))

        # Append new point and label
        points_dict[ann_frame_idx][self.ann_obj_id] = np.append(points_dict[ann_frame_idx][self.ann_obj_id], point, axis=0)
        labels_dict[ann_frame_idx][self.ann_obj_id] = np.append(labels_dict[ann_frame_idx][self.ann_obj_id], label, axis=0)

        self.algo_api.add_points(click_stack)
        save_click_stack(self, click_stack)
        masked_frame_with_markers = self.get_masked_frame(click_stack)
        return masked_frame_with_markers, click_stack

    def clean(self, reset_clicked_state, scale_slider, checkpoint, click_stack, input_first_frame, frame_per):
        if int(reset_clicked_state):
            self.algo_api.clean()
            input_video = self.video['segments'][self.segment_id]['path']
            output_paths, first_frame_rgb = get_meta_from_video(self, input_video, scale_slider)
            click_stack = ({}, {})
            self.frame_num = 0
            click_stack, num_frames = self.algo_api.initialize_sam(checkpoint, output_paths, click_stack)
            masked_frame = self.get_masked_frame(click_stack)
            return click_stack, masked_frame, gr.Slider(maximum=num_frames - 1, value=0)
        else:
            return click_stack, input_first_frame, frame_per

    def get_masked_frame(self, click_stack, mask_dir=None):
        image_path = os.path.join(self.video['video_metadata']['output_dir'], f'output_frames/{self.frame_num:07d}.jpg')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masked_frame = image.copy()

        if self.segmentation_state:
            if self.frame_num in self.algo_api.segment_masks:
                for obj_id in self.algo_api.segment_masks[self.frame_num]:
                    mask = self.algo_api.segment_masks[self.frame_num][obj_id]
                    masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)
                    if mask_dir is not None:
                        mask_output_path = os.path.join(mask_dir, f'{obj_id}_{self.frame_num:07d}.png')
                        cv2.imwrite(mask_output_path, masked_frame)

        # Always draw markers
        points_dict, labels_dict = click_stack
        if self.frame_num in points_dict and self.frame_num in labels_dict:
            masked_frame = draw_markers(masked_frame, points_dict[self.frame_num], labels_dict[self.frame_num])
        return masked_frame