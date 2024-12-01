import os
import gradio as gr
import yaml
import argparse
from java_functions import return_java_function
from backend import Backend


def seg_track_app(args):
    with open(os.path.join(os.path.dirname(__file__), 'config.yaml'), "r") as file:
        config = yaml.safe_load(file)  # safer alternative to yaml.load

    css = """
    #input_output_video video {
        max-height: 550px;
        max-width: 100%;
        height: auto;
    }
    """

    app = gr.Blocks(css=css)
    backend = Backend(config, args.base_dir)

    with app:
        gr.Markdown(
            '''
            <div style="text-align:center;">
                <span style="font-size:3em; font-weight:bold;">SAM2 for Video Segmentation 🔥</span>
            </div>
            This api supports using box (generated by scribble) and point prompts for video segmentation with SAM2.

            1. Upload video file 
            2. Select model size and downsample frame rate and run `Preprocess`
            3. Use `Stroke to Box Prompt` to draw box on the first frame or `Point Prompt` to click on the first frame
            4. Click `Segment` to get the segmentation result
            5. Click `Add New Object` to add new object
            6. Click `Start Tracking` to track objects in the video
            7. Click `Reset` to reset the app
            8. Download the video with segmentation result
            '''
        )

        click_stack = gr.State(({}, {}))
        frame_num = gr.State(value=(int(0)))
        last_draw = gr.State(None)

        with gr.Row():
            # Left column - make it wider with scale=0.7 (70% of width)
            with gr.Column(scale=1):
                with gr.Row():
                    # New large video upload section
                    with gr.Group():
                        large_video_input = gr.File(
                            label="Upload Large Video",
                            file_types=[".mp4", ".avi", ".mov"],
                            type="filepath"
                        )
                        upload_status = gr.Textbox(label="Upload Status", interactive=False)
                        with gr.Row():
                            prev_video_btn = gr.Button("← Previous")
                            video_index_slider = gr.Slider(
                                minimum=1,
                                maximum=1,
                                step=1,
                                value=1,
                                label="Video Segment",
                                interactive=True
                            )
                            next_video_btn = gr.Button("Next →")

                # Existing video input section
                with gr.Row():
                    tab_video_input = gr.Tab(label="Video input")
                    with tab_video_input:
                        seg_input_video = gr.Video(
                            label='Input video',
                            elem_id="input_output_video",
                        )
                        with gr.Row():
                            checkpoint = gr.Dropdown(label="Model Size",
                                                     choices=["tiny", "small", "base-plus", "large"], value="tiny")
                            scale_slider = gr.Slider(
                                label="Downsample Frame Rate",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.25,
                                value=1.0,
                                interactive=True
                            )
                            preprocess_button = gr.Button(value="Preprocess", interactive=True)

                with gr.Row():
                    # Put Point Prompt tab first and set selected=True
                    tab_click = gr.Tab(label="Point Prompt")
                    with tab_click:
                        with gr.Row():
                            zoom_in = gr.Button("🔍 Zoom In")
                            zoom_out = gr.Button("🔍 Zoom Out")
                            reset_zoom = gr.Button("↺ Reset View")
                            toggle_seg = gr.Button("👁 Toggle Segmentation")

                        input_first_frame = gr.Image(
                            label='Segmentation results',
                            interactive=True,
                            show_download_button=True,
                            show_label=True,
                            height=700,
                            type="numpy"
                        )
                        with gr.Row():
                            with gr.Column(scale=1):
                                point_mode = gr.Radio(
                                    choices=["Positive", "Negative"],
                                    value="Positive",
                                    label="Point Prompt",
                                    interactive=True
                                )
                            with gr.Column(scale=1):
                                undo_point = gr.Button(
                                    value="Undo Last Point",
                                    interactive=True
                                )

                    # Put Stroke to Box Prompt tab second
                    tab_stroke = gr.Tab(label="Stroke to Box Prompt")
                    with tab_stroke:
                        drawing_board = gr.Sketchpad(
                            label="Drawing Board",
                            interactive=True,
                            height=700,
                            width=700
                        )
                        with gr.Row():
                            seg_acc_stroke = gr.Button(value="Segment", interactive=True)

                with gr.Row():
                    with gr.Column():
                        frame_per = gr.Slider(
                            label="Percentage of Frames Viewed",
                            minimum=0.0,
                            maximum=100.0,
                            step=1.0,
                            value=0.0,
                        )
                        new_object_button = gr.Button(
                            value="Add New Object",
                            interactive=True
                        )
                        track_for_video = gr.Button(
                            value="Start Tracking",
                            interactive=True,
                        )
                        reset_button = gr.Button(
                            value="Reset",
                            interactive=True,
                        )

        gr.Markdown(
            '''
            <div style="text-align:center; margin-top: 20px;">
                The authors of this work highly appreciate Meta AI for making SAM2 publicly available to the community. 
                The interface was built on <a href="https://github.com/z-x-yang/Segment-and-Track-Anything/blob/main/tutorial/tutorial%20for%20WebUI-1.0-Version.md" target="_blank">SegTracker</a>. 
                <a href="https://docs.google.com/document/d/1idDBV0faOjdjVs-iAHr0uSrw_9_ZzLGrUI2FEdK-lso/edit?usp=sharing" target="_blank">Data Source</a>.
            </div>
                '''
        )
        frame_per.change(fn=None, inputs=[], outputs=[],
                         js=return_java_function(java_input='frame_per'))

        frame_per.change(
            fn=backend.move_slider,
            inputs=[frame_per, click_stack],
            outputs=[input_first_frame, frame_num]
        )

        # Listen to the preprocess button click to get the first frame of video with scaling
        preprocess_button.click(
            fn=backend.preprocess_video,
            inputs=[scale_slider, checkpoint],
            outputs=[click_stack, input_first_frame, frame_per]
        )

        # Interactively modify the mask acc click
        input_first_frame.select(
            fn=backend.sam_click,
            inputs=[frame_num, point_mode, click_stack],
            outputs=[input_first_frame, click_stack]
        )

        # Track object in video
        track_for_video.click(
            fn=backend.tracking_objects,
            inputs=[frame_num, click_stack],
            outputs=[input_first_frame, drawing_board]
        )

        # Confirm Dialog
        reset_clicked_state = gr.Textbox(placeholder="subject", visible=False)
        reset_button.click(None, inputs=[], outputs=reset_clicked_state,
                           js=return_java_function(java_input='reset_dialog'))

        reset_clicked_state.change(
            fn=backend.clean,
            inputs=[reset_clicked_state, scale_slider, checkpoint, click_stack, input_first_frame, frame_per],
            outputs=[click_stack, input_first_frame, frame_per]
        )

        new_object_button.click(
            fn=backend.increment_ann_obj_id,
            inputs=[],
            outputs=[]
        )

        tab_stroke.select(
            fn=backend.drawing_board_get_input_first_frame,
            inputs=[input_first_frame],
            outputs=[drawing_board],
        )

        seg_acc_stroke.click(
            fn=backend.sam_stroke,
            inputs=[drawing_board, last_draw, frame_num],
            outputs=[input_first_frame, drawing_board, last_draw]
        )

        undo_point.click(
            fn=backend.undo_last_point,
            inputs=[frame_num, click_stack],
            outputs=[input_first_frame, click_stack]
        )

        # Add JavaScript for zoom functionality
        zoom_in.click(fn=None, inputs=[], outputs=[], js=return_java_function(java_input='zoom_in'))
        zoom_out.click(fn=None, inputs=[], outputs=[], js=return_java_function(java_input='zoom_out'))
        reset_zoom.click(fn=None, inputs=[], outputs=[], js=return_java_function(java_input='reset_zoom'))

        # Add keyboard shortcuts
        app.load(js=return_java_function(java_input='keyboard_shortcuts'))

        # Add this JavaScript to handle image dragging
        app.load(js=return_java_function(java_input='image_dragging'))

        toggle_seg.click(
            fn=backend.toggle_segmentation,
            inputs=[frame_num, click_stack],
            outputs=[input_first_frame, drawing_board, click_stack]
        )

        # Event handlers
        large_video_input.upload(
            fn=backend.handle_large_video_upload,
            inputs=[large_video_input],
            outputs=[upload_status, seg_input_video, video_index_slider],
        )

        upload_status.change(None, inputs=[upload_status], outputs=[],
                             js=return_java_function(java_input='generic_dialog'))

        video_index_slider.change(
            fn=backend.load_video_segment,
            inputs=[video_index_slider],
            outputs=[seg_input_video],
            api_name="slider_change"  # Add this for debugging
        )

        next_video_btn.click(
            fn=backend.increment_video_index,
            inputs=[video_index_slider],
            outputs=[video_index_slider]
        )

        prev_video_btn.click(
            fn=backend.decrement_video_index,
            inputs=[video_index_slider],
            outputs=[video_index_slider]
        )

    app.queue()
    app.launch(
        debug=True,
        share=True,
        server_port=7860,
        server_name="0.0.0.0",
        # Use app_kwargs to pass FastAPI configuration
        app_kwargs={
            "max_request_size": 1024 * 1024 * 51200,  # Increase to 50GB
            "timeout": 60 * 10,  # Increase timeout to 10 minutes
            "max_upload_size": 1024 * 1024 * 51200  # Also set max upload size
        }
    )

def get_args():
    parser = argparse.ArgumentParser(description="Process a file path.")
    parser.add_argument('base_dir', type=str, default=os.path.join(os.getcwd(), 'annotation'),
                        nargs='?', help="Path to the file")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    seg_track_app(get_args())