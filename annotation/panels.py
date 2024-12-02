import gradio as gr


def patient_panel():
    panel = {}
    with gr.Accordion("Patient Panel", open=True):
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

    return panel, submit


def tissue_panel():
    panel = {}
    with gr.Accordion("Tissue Panel", open=True):
        with gr.Row():
            panel['filter'] = gr.Dropdown(["WLI", "LCI", "BLI"], label="Filter")
            panel['perspective'] = gr.Dropdown(["Micro", "Medium", "Macro", "Unclear"], label="Perspective")
            panel['tumor_type'] = gr.Dropdown(['SCC', 'Adenocarcinoma', 'Normal_Tissue', 'Adenoma', 'Other'],
                                     label="Tumor Type")
            panel['indigo_carmine'] = gr.Checkbox(label="Indigo Carmine (the blue stuff)")
        with gr.Row():
            panel['good_image'] = gr.Checkbox(label="Good Image")
            panel['indication'] = gr.Dropdown(["Colon", "Stomach", "Esophagus", "Other", "Unknown"], label="Indication")
            panel['site'] = gr.Dropdown([
                'Body', 'Terminal ileum', 'Transverse', 'Antrum', 'Pre-pyloric', 'Sigmoid ', 'Ascending', 'IU4549',
                'Colon', 'Cecum', 'Sigmoid', 'Ascending ', 'Rectum', 'Anastomosis', 'Descending',
                'Ileo-cecal valve', 'NaN', 'Middle third', 'Fundus', 'Lower third', 'Incisura',
                'Cardia', 'Upper third', 'Other', 'Unknown'
            ], label="Site")
            panel['approved'] = gr.Checkbox(label="Approved")
        with gr.Row():
            panel['health_status'] = gr.Dropdown(["Healthy", "Other"], label="Healthy/Other")
            panel['comment'] = gr.Textbox(label="Comment", placeholder="Add any comments here...")

    return panel