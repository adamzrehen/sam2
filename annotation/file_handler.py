import os
import shutil
import cv2
import ffmpeg
import pickle


def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)


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


def get_meta_from_video(self, input_video, scale_slider):
    # Verify upload first
    verified_path = verify_upload(input_video)
    if verified_path is None:
        print("Video upload verification failed")
        return None, ({}, {}), None, None, 0, None, None, None, 0

    output_keys = ['frames_dir', 'masks_dir', 'combined_dir']
    output_paths = {key: os.path.join(self.base_dir, self.video['video_metadata']['output_dir'], self.config[key])
                    for key in output_keys}
    for key in ['frames_dir', 'masks_dir', 'combined_dir']:
        clear_folder(output_paths[key])

    if input_video is None:
        return None, ({}, {}), None, None, 0, None, None, None, 0
    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert total_frames > 0, "No frames found in the input video"
    cap.release()
    output_frames = int(total_frames * scale_slider)
    frame_interval = max(1, total_frames // output_frames)
    ffmpeg.input(input_video, hwaccel='cuda').output(
        os.path.join(output_paths['frames_dir'], '%07d.jpg'),
        q=2,
        start_number=0,
        vf=rf'select=not(mod(n\,{frame_interval}))',
        vsync='vfr'
    ).run()

    first_frame_path = os.path.join(output_paths['frames_dir'], '0000000.jpg')
    first_frame = cv2.imread(first_frame_path)
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    return output_paths, first_frame_rgb


def save_click_stack(self, click_stack):
    segment_name = os.path.basename(self.video['segments'][self.segment_id]['path'].split('.mp4')[0])
    output_path = os.path.join(self.base_dir, 'data/sam2', self.video['video_metadata']['video_name'],
                               f'segments/click_stack_{segment_name}' + '.pkl')
    with open(output_path, 'wb') as file:
        pickle.dump(click_stack, file)


def load_click_stack(self):
    segment_name = os.path.basename(self.video['segments'][self.segment_id]['path'].split('.mp4')[0])
    inference_state_path = os.path.join(self.base_dir,
                                        os.path.dirname(self.video['segments'][self.segment_id]['path']),
                                        f'click_stack_{segment_name}.pkl')
    click_stack = ({}, {})
    if os.path.exists(inference_state_path):
        segment_name = os.path.basename(self.video['segments'][self.segment_id]['path'].split('.mp4')[0])
        input_path = os.path.join(self.base_dir, 'data/sam2', self.video['video_metadata']['video_name'],
                                  f'segments/click_stack_{segment_name}' + '.pkl')
        with open(input_path, 'rb') as file:
            click_stack = pickle.load(file)
    return click_stack


def load_existing_video_metadata(self, video_metadata):
    try:
        with open(os.path.join(self.base_dir, video_metadata["output_dir"],
                                   f'{video_metadata["video_name"]}_meta.pkl'), 'rb') as file:
            self.video = pickle.load(file)
        return True
    except Exception as e:
        return False