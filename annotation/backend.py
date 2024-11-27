import os
import cv2
import math
import shutil
import gradio as gr
import ffmpeg
import numpy as np
import pickle
from utils import clear_folder, draw_markers, show_mask, zip_folder
from algo_api import AlgoAPI


class Backend:
    def __init__(self):
        self.base_dir = os.path.join(os.getcwd(), 'annotation')
        self.algo_api = AlgoAPI(base_dir=self.base_dir)
        self.video = {}

    @staticmethod
    def show_res_by_slider(frame_per, click_stack):
        image_path = 'output_frames'
        output_combined_dir = 'output_combined'

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

    def tracking_objects(self, seg_tracker, frame_num, input_video):
        output_dir = 'output_frames'
        output_masks_dir = 'output_masks'
        output_combined_dir = 'output_combined'
        output_video_path = 'output_video.mp4'
        output_zip_path = 'output_masks.zip'
        clear_folder(output_masks_dir)
        clear_folder(output_combined_dir)
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        if os.path.exists(output_zip_path):
            os.remove(output_zip_path)
        video_segments = {}
        predictor, inference_state, image_predictor = seg_tracker
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
        # for frame_idx in sorted(video_segments.keys()):
        for frame_file in frame_files:
            frame_idx = int(os.path.splitext(frame_file)[0])
            frame_path = os.path.join(output_dir, frame_file)
            image = cv2.imread(frame_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masked_frame = image.copy()
            if frame_idx in video_segments:
                for obj_id, mask in video_segments[frame_idx].items():
                    masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)
                    mask_output_path = os.path.join(output_masks_dir, f'{obj_id}_{frame_idx:07d}.png')
                    cv2.imwrite(mask_output_path, show_mask(mask))
            combined_output_path = os.path.join(output_combined_dir, f'{frame_idx:07d}.png')
            combined_image_bgr = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(combined_output_path, combined_image_bgr)
            if frame_idx == frame_num:
                final_masked_frame = masked_frame

        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # output_frames = int(total_frames * scale_slider)
        output_frames = len([name for name in os.listdir(output_combined_dir) if
                             os.path.isfile(os.path.join(output_combined_dir, name)) and name.endswith('.png')])
        out_fps = fps * output_frames / total_frames
        ffmpeg.input(os.path.join(output_combined_dir, '%07d.png'), framerate=out_fps).output(output_video_path,
                                                                                              vcodec='h264_nvenc',
                                                                                              pix_fmt='yuv420p').run()
        zip_folder(output_masks_dir, output_zip_path)
        print("done")
        return final_masked_frame, final_masked_frame, output_video_path, output_video_path, output_zip_path

    @staticmethod
    def increment_ann_obj_id(ann_obj_id):
        ann_obj_id += 1
        return ann_obj_id

    @staticmethod
    def drawing_board_get_input_first_frame(input_first_frame):
        return input_first_frame

    @staticmethod
    def undo_last_point(seg_tracker, frame_num, click_stack):
        points_dict, labels_dict = click_stack

        if frame_num in points_dict and frame_num in labels_dict:
            for obj_id in points_dict[frame_num]:
                if len(points_dict[frame_num][obj_id]) > 0:
                    # Remove the last point and its corresponding label
                    points_dict[frame_num][obj_id] = points_dict[frame_num][obj_id][:-1]
                    labels_dict[frame_num][obj_id] = labels_dict[frame_num][obj_id][:-1]

        # Redraw the frame with updated points
        image_path = f'output_frames/{frame_num:07d}.jpg'
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masked_frame = image.copy()
        # Redraw existing masks if any
        predictor, inference_state, image_predictor = seg_tracker
        if frame_num in points_dict and points_dict[frame_num]:
            for obj_id in points_dict[frame_num]:
                if len(points_dict[frame_num][obj_id]) > 0:
                    frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points(
                        inference_state=inference_state,
                        frame_idx=frame_num,
                        obj_id=obj_id,
                        points=points_dict[frame_num][obj_id],
                        labels=labels_dict[frame_num][obj_id],
                    )
                    for i, obj_id in enumerate(out_obj_ids):
                        mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                        masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)

        if frame_num in points_dict and frame_num in labels_dict:
            masked_frame = draw_markers(masked_frame, points_dict[frame_num], labels_dict[frame_num])

        return seg_tracker, masked_frame, masked_frame, (points_dict, labels_dict)

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

    def toggle_segmentation(self, seg_tracker, frame_num, click_stack):
        points_dict, labels_dict = click_stack
        predictor, inference_state, image_predictor = seg_tracker

        # Load the original image
        image_path = f'output_frames/{frame_num:07d}.jpg'
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get current state (with or without segmentation)
        current_state = getattr(self.toggle_segmentation, 'show_seg', True)

        if current_state:
            # Return clean image but keep points visible
            masked_frame = image.copy()
            if frame_num in points_dict:
                masked_frame = draw_markers(masked_frame, points_dict[frame_num], labels_dict[frame_num])
        else:
            # Return image with segmentation
            masked_frame = image.copy()
            if frame_num in points_dict and frame_num in labels_dict:
                try:
                    # Try to get the last valid inference state
                    for obj_id in points_dict[frame_num]:
                        if len(points_dict[frame_num][obj_id]) > 0:
                            # Store the current inference state
                            current_inference_state = inference_state

                            frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points(
                                inference_state=current_inference_state,
                                frame_idx=frame_num,
                                obj_id=obj_id,
                                points=points_dict[frame_num][obj_id],
                                labels=labels_dict[frame_num][obj_id],
                            )

                            # Only update if we got valid masks
                            if out_mask_logits is not None and len(out_mask_logits) > 0:
                                for i, obj_id in enumerate(out_obj_ids):
                                    mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                                    masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)

                    # Always draw markers
                    masked_frame = self.draw_markers(masked_frame, points_dict[frame_num], labels_dict[frame_num])

                except Exception as e:
                    print(f"Error during segmentation: {e}")
                    # If segmentation fails, at least show the points
                    masked_frame = self.draw_markers(masked_frame, points_dict[frame_num], labels_dict[frame_num])

        # Toggle state for next click
        self.toggle_segmentation.show_seg = not current_state
        return seg_tracker, masked_frame, masked_frame, click_stack

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
            video_metadata["output_dir"] = os.path.join('data', 'sam2', video_metadata["video_name"])
            if os.path.exists(video_metadata["output_dir"]):
                 status = self.load_existing_video(video_metadata)
            if not status:
                os.makedirs(video_metadata["output_dir"])

                # move file to output directory
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

    def load_existing_video(self, video_metadata):
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

    def sam_stroke(self, seg_tracker, drawing_board, last_draw, frame_num, ann_obj_id):
        return self.algo_api.sam_stroke(seg_tracker, drawing_board, last_draw, frame_num, ann_obj_id)

    def get_meta_from_video(self, seg_tracker, input_video, scale_slider, checkpoint):
        return self.algo_api.get_meta_from_video(seg_tracker, input_video, scale_slider, checkpoint)

    def sam_click(self, seg_tracker, frame_num, point_mode, click_stack, ann_obj_id, evt: gr.SelectData):
        return self.algo_api.sam_click(seg_tracker, frame_num, point_mode, click_stack, ann_obj_id, evt)

    def clean(self, seg_tracker):
        return self.algo_api.clean(seg_tracker)