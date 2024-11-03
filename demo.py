# Importing necessary libraries
"""
Image Color Extraction Tool

Author: Ali Bahrami
GitHub: https://github.com/alibahrami2001
Date: 11/3/2024
"""
import io
import gradio as gr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image

from image_color_extraction import ImageColorExtractor, visualize_colors

image_color_extractor = ImageColorExtractor(each_channels_possible_values=8)


def rgb_to_hex(r, g, b):
    return "#" + ("{:02X}" * 3).format(r, g, b)


def get_extracted_colors_df(extracted_colors):
    hex_colors = []
    for color in extracted_colors:
        hex_colors.append(rgb_to_hex(*np.floor(color * 255).astype(np.uint8).tolist()))

    extracted_colors_df = pd.DataFrame({"hex": hex_colors})

    return extracted_colors_df


def run_handler(image, max_num_colors):
    max_image_size = 1024
    image.thumbnail((max_image_size, max_image_size), Image.Resampling.LANCZOS)
    extracted_colors = image_color_extractor(image, max_num_colors=max_num_colors)
    visualized_colors = visualize_colors(extracted_colors)
    extracted_colors_df = get_extracted_colors_df(extracted_colors)
    return visualized_colors, extracted_colors_df


with gr.Blocks(gr.themes.Ocean()) as demo:
    with gr.Row():
        gr.HTML("""
    <h1 style='
        text-align: center; 
        background: linear-gradient(90deg, orange, teal, navy); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5);
        font-size: 3em;'>
        Image Color Extraction Tool
    </h1>
""")
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", interactive=True, label="Input Image")
            max_num_colors = gr.Slider(
                minimum=1, maximum=256, value=16, label="Max Number of Colors"
            )
            run_btn = gr.Button("Run")
        with gr.Column():
            extracted_colors_image = gr.Image(
                type="pil", format="png", label="Extracted Colors Visualization"
            )
            extracted_colors_rgb = gr.DataFrame()

    run_btn.click(
        fn=run_handler,
        inputs=[image, max_num_colors],
        outputs=[extracted_colors_image, extracted_colors_rgb],
    )

demo.launch()
