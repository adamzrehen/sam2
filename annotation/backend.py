import os
import cv2
import math
import shutil
import gradio as gr
import ffmpeg
import numpy as np
import pickle
from utils import draw_markers, show_mask, zip_folder
from file_handler import clear_folder, get_meta_from_video
from algo_api import AlgoAPI


class Backend:
    def __init__(self, config):
        self.base_dir = os.path.join(os.getcwd(), 'annotation')
        self.config = config
        self.algo_api = AlgoAPI(base_dir=self.base_dir, config=self.config)
        self.video = {}
        self.segmentation_state = False

    def show_res_by_slider(self, frame_per, click_stack):
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
            chosen_frame_path = output_masked_frame_path[frame_per]
            print(f"{chosen_frame_path}")
            chosen_frame_show = cv2.imread(chosen_frame_path)
            chosen_frame_show = cv2.cvtColor(chosen_frame_show, cv2.COLOR_BGR2RGB)
            points_dict, labels_dict = click_stack
            if frame_per in points_dict and frame_per in labels_dict:
                chosen_frame_show = draw_markers(chosen_frame_show, points_dict[frame_per], labels_dict[frame_per])
            return chosen_frame_show, chosen_frame_show, frame_per

    def tracking_objects(self, frame_num, input_video):
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


        predictor, inference_state, image_predictor = self.seg_tracker
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            self.algo_api.segment_masks[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        frame_files = sorted([f for f in os.listdir(output_paths['frames_dir']) if f.endswith('.jpg')])
        # for frame_idx in sorted(video_segments.keys()):
        for frame_file in frame_files:
            frame_idx = int(os.path.splitext(frame_file)[0])
            frame_path = os.path.join(output_paths['frames_dir'], frame_file)
            image = cv2.imread(frame_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masked_frame = image.copy()
            if frame_idx in self.algo_api.segment_masks:
                for obj_id, mask in self.algo_api.segment_masks[frame_idx].items():
                    masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)
                    mask_output_path = os.path.join(output_paths['masks_dir'], f'{obj_id}_{frame_idx:07d}.png')
                    cv2.imwrite(mask_output_path, show_mask(mask))
            combined_output_path = os.path.join(output_paths['combined_dir'], f'{frame_idx:07d}.png')
            combined_image_bgr = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(combined_output_path, combined_image_bgr)
            if frame_idx == frame_num:
                final_masked_frame = masked_frame

        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # output_frames = int(total_frames * scale_slider)
        output_frames = len([name for name in os.listdir(output_paths['combined_dir']) if
                             os.path.isfile(os.path.join(output_paths['combined_dir'], name)) and name.endswith('.png')])
        out_fps = fps * output_frames / total_frames
        ffmpeg.input(os.path.join(output_paths['combined_dir'], '%07d.png'), framerate=out_fps).output(output_paths['video_path'],
                                                                                              vcodec='h264_nvenc',
                                                                                              pix_fmt='yuv420p').run()
        zip_folder(output_paths['masks_dir'], output_paths['zip_path'])
        return (final_masked_frame, final_masked_frame, output_paths['video_path'], output_paths['video_path'],
                output_paths['zip_path'])

    @staticmethod
    def increment_ann_obj_id(ann_obj_id):
        ann_obj_id += 1
        return ann_obj_id

    @staticmethod
    def drawing_board_get_input_first_frame(input_first_frame):
        return input_first_frame

    def undo_last_point(self, frame_num, click_stack):
        points_dict, labels_dict = click_stack

        # Remove the last point and its corresponding label
        if frame_num in points_dict and frame_num in labels_dict:
            for obj_id in points_dict[frame_num]:
                if len(points_dict[frame_num][obj_id]) > 0:
                    points_dict[frame_num][obj_id] = points_dict[frame_num][obj_id][:-1]
                    labels_dict[frame_num][obj_id] = labels_dict[frame_num][obj_id][:-1]

        # Redraw the frame with updated points
        image_path = os.path.join(self.video['video_metadata']['output_dir'], f'output_frames/{frame_num:07d}.jpg')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Redraw existing masks if any
        masked_frame = image.copy()
        self.algo_api.add_points((points_dict, labels_dict))
        if frame_num in self.algo_api.segment_masks:
            for obj_id in self.algo_api.segment_masks[frame_num]:
                mask = self.algo_api.segment_masks[frame_num][obj_id]
                masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)

        if frame_num in points_dict and frame_num in labels_dict:
            masked_frame = draw_markers(masked_frame, points_dict[frame_num], labels_dict[frame_num])

        return masked_frame, masked_frame, (points_dict, labels_dict)

    @staticmethod
    def zoom_to_last_point(frame_num, click_stack):
        points_dict, labels_dict = click_stack

        # Get current frame points
        if frame_num in points_dict and frame_num in labels_dict:
            # Find the last positive point across all objects
            last_point = None
            for obj_id in points_dict[frame_num]:
                points = points_dict[frame_num][obj_id]
                labels = labels_dict[frame_num][obj_id]
                if len(points) > 0 and len(labels) > 0:
                    # Find positive points (label == 1)
                    positive_indices = np.where(labels == 1)[0]
                    if len(positive_indices) > 0:
                        last_positive_idx = positive_indices[-1]
                        last_point = points[last_positive_idx]

        if last_point is not None:
            x, y = last_point[0], last_point[1]
        else:
            image_path = f'output_frames/{frame_num:07d}.jpg'
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            x, y = w / 2, h / 2
        return x, y

    def toggle_segmentation(self, frame_num, click_stack):
        points_dict, labels_dict = click_stack

        # Load the original image
        image_path = os.path.join(self.video['video_metadata']['output_dir'], f'output_frames/{frame_num:07d}.jpg')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.segmentation_state:
            # Return clean image but keep points visible
            masked_frame = image.copy()
            if frame_num in points_dict:
                masked_frame = draw_markers(masked_frame, points_dict[frame_num], labels_dict[frame_num])
        else:
            # Return image with segmentation
            masked_frame = image.copy()
            if frame_num in self.algo_api.segment_masks:
                for obj_id in self.algo_api.segment_masks[frame_num]:
                    mask = self.algo_api.segment_masks[frame_num][obj_id]
                    masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)

            # Always draw markers
            if frame_num in points_dict and frame_num in labels_dict:
                masked_frame = draw_markers(masked_frame, points_dict[frame_num], labels_dict[frame_num])

        # Toggle state for next click
        self.segmentation_state = not self.segmentation_state
        return masked_frame, masked_frame, click_stack

    @staticmethod
    def split_video(video_path, video_name, output_dir, progress_callback=None, max_size_mb=20, max_frames=100):
        """
        Splits a video into smaller segments based on size or number of frames.
        """
        # Get video directory name from input path
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
        """Handle upload and splitting of large video file"""
        if file_path is None:
            return "No file uploaded", None, gr.Slider.update(), None
        if not os.path.exists(file_path.name):
            return f"ERROR File {file_path.name} does not exist", None, gr.Slider.update(), None
        file_size = os.path.getsize(file_path.name)
        if file_size < 0.2 * 1024 * 1024:
            return f"ERROR File {file_path.name} is too small", None, gr.Slider.update(), None

        try:
            # Get original filename
            video_metadata = {"video_name": os.path.splitext(os.path.basename(file_path.name))[0]}

            # set output directory
            video_metadata["output_dir"] = os.path.join(self.base_dir, 'data', 'sam2', video_metadata["video_name"])

            status = False
            if os.path.exists(video_metadata["output_dir"]):
                 status = self.load_existing_video_metadata(video_metadata)
            if not status:
                os.makedirs(video_metadata["output_dir"])

                # Move file to output directory
                video_metadata["original_video_path"] = os.path.join(video_metadata["output_dir"],
                                                                     os.path.basename(file_path.name))
                shutil.move(file_path.name, video_metadata["original_video_path"])

                if not os.path.exists(video_metadata["original_video_path"]):
                    return (f'ERROR Failed to move file {file_path.name} to {video_metadata["original_video_path"]}',
                            None, gr.Slider.update(), None)

                yield f"Upload complete. Processing video...", None, gr.Slider.update(), video_metadata
                # Get initial video info
                try:
                    video_metadata["probe"] = ffmpeg.probe(video_metadata["original_video_path"])
                except Exception as e:
                    print(f"Error getting video info: {e}")

                video_metadata["video_info"] = next(s for s in video_metadata["probe"]['streams'] if s['codec_type'] == 'video')
                video_metadata["total_frames"] = int(video_metadata["video_info"]['nb_frames'])

                segment_infos, num_segments, metadata = self.split_video(
                    video_metadata["original_video_path"], video_metadata["video_name"], video_metadata["output_dir"],
                    progress_callback=lambda x: (f"Splitting video: {x:.1f}%", None, gr.Slider.update(), None)
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
                self.algo_api.update_video(self.video)
                yield (
                    "Processing complete. Select a segment to begin.",
                    self.video['segments'][0]['path'],
                    gr.Slider.update(maximum=self.video['metadata']['segments_created'], value=1),
                    self.video['metadata']
                )
            else:
                yield "Error: Failed to split video.", None, gr.Slider.update(), None

        except Exception as e:
            print(f"Error processing upload: {e}")
            yield f"Error during upload: {str(e)}", None, gr.Slider.update(), None

    def load_existing_video_metadata(self, video_metadata):
        try:
            with open(os.path.join(self.base_dir, video_metadata["output_dir"],
                                       f'{video_metadata["video_name"]}_meta.pkl'), 'rb') as file:
                self.video = pickle.load(file)
            return True
        except Exception as e:
            return False

    def load_video_segment(self, index):
        """Load a specific segment"""
        print("DEBUG: load_video_segment called!")  # Basic debug print
        print(f"DEBUG: index={index}, segments={self.video['paths']}")
        self.algo_api.update_segment_id(index)
        return self.video['paths'][index - 1]

    def increment_video_index(self, current_index):
        """Increment video index while staying within bounds"""
        print(f"Current index: {current_index}, Segments: {self.video['paths']}")  # Debug print
        max_index = len(self.video['paths']) if self.video['paths'] else 1
        return min(max_index, current_index + 1)

    @staticmethod
    def decrement_video_index(current_index):
        """Decrement video index while staying within bounds"""
        return max(1, current_index - 1)

    def sam_stroke(self, drawing_board, last_draw, frame_num, ann_obj_id):
        return self.algo_api.sam_stroke(drawing_board, last_draw, frame_num, ann_obj_id)

    def get_meta_from_video(self, input_video, scale_slider, checkpoint):
        output_paths, first_frame_rgb = get_meta_from_video(self, input_video, scale_slider)
        click_stack, num_frames = self.algo_api.initialize_sam(checkpoint, output_paths)
        return (click_stack, first_frame_rgb, first_frame_rgb, None, None, None, 0,
                gr.Slider.update(maximum=num_frames, value=0))

    def sam_click(self, frame_num, point_mode, click_stack, ann_obj_id, evt: gr.SelectData):
        return self.algo_api.sam_click(frame_num, point_mode, click_stack, ann_obj_id, evt)

    def clean(self):
        return self.algo_api.clean()