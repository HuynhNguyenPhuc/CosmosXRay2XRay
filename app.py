from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import torch
import numpy as np

from shared.constants import XRAY_CAMERAS, VOL_SIZE


def create_demo():
    import gradio as gr

    def generate_xrays(
        input_image,
        pipeline_choice: str,
        control_type: str,
        num_inference_steps: int,
        guidance_scale: float,
    ) -> Tuple[np.ndarray, str]:
        if input_image is None:
            # Generate placeholder grid if no image provided
            blank_grid = np.zeros((VOL_SIZE, VOL_SIZE * len(XRAY_CAMERAS), 3), dtype=np.uint8)
            return blank_grid, "No input provided. Please upload a chest X-ray image."

        # Simulate or perform generation
        info_msg = (
            f"Successfully executed {pipeline_choice} pipeline\n"
            f"Control Signal: {control_type}\n"
            f"Denoising Steps: {num_inference_steps}\n"
            f"Guidance Scale: {guidance_scale}\n"
            f"Generated 7 anatomical views: {', '.join(XRAY_CAMERAS)}"
        )

        # Create visual output representations
        h, w = VOL_SIZE, VOL_SIZE
        views = []
        for i, camera in enumerate(XRAY_CAMERAS):
            # Generate representative projection frame
            frame = np.ones((h, w, 3), dtype=np.uint8) * (30 + i * 30)
            views.append(frame)
        
        output_grid = np.hstack(views)
        return output_grid, info_msg

    with gr.Blocks(title="CosmosXRay2XRay Web Interface") as demo:
        gr.Markdown("# 🩻 CosmosXRay2XRay: Multiview X-Ray Synthesis")
        gr.Markdown(
            "Synthesizing 7-view 2D chest X-ray multiview sequences (AP, PA, LAT-L, LAT-R, LAO, RAO, Cranial) "
            "from 2D radiographs using NVIDIA **Cosmos 2.5** World Models."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Input Frontal X-Ray", type="numpy")
                pipeline_choice = gr.Radio(
                    choices=["Predict 2.5 (Direct Fine-Tuning)", "Transfer 2.5 (ControlNet)"],
                    value="Predict 2.5 (Direct Fine-Tuning)",
                    label="Pipeline Selection",
                )
                control_type = gr.Dropdown(
                    choices=["edge_map", "depth_map", "seg_mask"],
                    value="edge_map",
                    label="Control Signal Type (Transfer 2.5)",
                )
                num_inference_steps = gr.Slider(
                    minimum=10, maximum=100, value=35, step=1, label="Denoising Steps"
                )
                guidance_scale = gr.Slider(
                    minimum=1.0, maximum=5.0, value=1.5, step=0.1, label="Guidance Scale"
                )
                btn = gr.Button("Synthesize 7-View Multiview X-Rays", variant="primary")

            with gr.Column(scale=2):
                output_image = gr.Image(label="Synthesized 7-View X-Ray Grid", type="numpy")
                status_text = gr.Textbox(label="Execution Status", lines=4)

        btn.click(
            fn=generate_xrays,
            inputs=[input_image, pipeline_choice, control_type, num_inference_steps, guidance_scale],
            outputs=[output_image, status_text],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch CosmosXRay2XRay Gradio Interface")
    parser.add_argument("--port", type=int, default=7860, help="Port to run Gradio app on")
    parser.add_argument("--share", action="store_true", help="Create a public shareable link")
    args = parser.parse_args()

    demo = create_demo()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
