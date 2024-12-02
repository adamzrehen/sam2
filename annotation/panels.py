import gradio as gr


def patient_panel():
    panel = {}
    with gr.Accordion("Input Panel", open=True):
        with gr.Row():
            panel['tagging_status'] = gr.Dropdown(["To tag", "On process", "For review", "Approved", "To fix"],
                                         label="Tagging Status")
            panel['camera'] = gr.Dropdown(["Fujifilm", "Olympus", "Other", "Unknown"], label="Camera")
        with gr.Row():
            panel['annotator'] = gr.Textbox(label="Annotator")
            panel['patient_code'] = gr.Textbox(label="Patient’s Code")
        with gr.Row():
            panel['gender'] = gr.Dropdown(["F", "M", "Unknown", "Other", "Non-binary", "Fluid", "Do not want to confirm"],
                                 label="Gender")
            panel['age'] = gr.Number(label="Age (years)", precision=0)
        with gr.Row():
            panel['height'] = gr.Number(label="Dimension Height (mm)", precision=2)
            panel['width'] = gr.Number(label="Dimension Width (mm)", precision=2)
            panel['thickness'] = gr.Number(label="Thickness (mm)", precision=2)
        with gr.Row():
            panel['tumor_height'] = gr.Number(label="Tumor Height (mm)", precision=2)
            panel['tumor_width'] = gr.Number(label="Tumor Width (mm)", precision=2)
            panel['depth'] = gr.Number(label="Tumor Depth (mm)", precision=2)
        with gr.Row():
            panel['grade'] = gr.Dropdown([], label="Grade")
            panel['stage'] = gr.Dropdown([], label="Stage")
        with gr.Row():
            panel['lvi'] = gr.Checkbox(label="LVI")
            panel['pni'] = gr.Checkbox(label="PNI")
        with gr.Row():
            panel['deep_superficial'] = gr.Dropdown(["Deep", "Superficial", "Unknown"], label="Deep / Superficial")
            panel['tissue_layer_invasion'] = gr.Dropdown([], label="Tissue Layer Invasion")

    with gr.Row():
        submit = gr.Button("Submit")
    with gr.Row():
        output = gr.Textbox(label="Output", interactive=False)

    return panel, submit, output
