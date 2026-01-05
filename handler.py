"""
RunPod Serverless Handler for Infinity-Parser-7B.

Converts scanned documents to Markdown using the Infinity-Parser model.
Excellent for Chinese documents, tables, and complex layouts.

Based on: https://huggingface.co/infly/Infinity-Parser-7B
"""

import base64
import os
import tempfile
import traceback
from pathlib import Path

import runpod
from runpod import RunPodLogger

# Initialize logger
log = RunPodLogger()

# Initialize model once at startup
log.info("Loading Infinity-Parser-7B model...")

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pdf2image import convert_from_path

MODEL_PATH = "infly/Infinity-Parser-7B"
PROMPT = "Please transform the document's contents into Markdown format."

# Load model with SDPA (PyTorch native attention - no compilation needed)
MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,  # Changed from torch_dtype (deprecated)
    attn_implementation="sdpa",  # Uses PyTorch's ScaledDotProductAttention
    device_map="auto",
)

# Configure processor with recommended pixel range
min_pixels = 256 * 28 * 28   # 448 x 448
max_pixels = 2304 * 28 * 28  # 1344 x 1344
PROCESSOR = AutoProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=min_pixels,
    max_pixels=max_pixels
)

log.info("Infinity-Parser-7B loaded successfully!")


def process_image(image_path: str) -> str:
    """Process a single image and return Markdown."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = PROCESSOR(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = MODEL.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = PROCESSOR.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0] if output_text else ""


def handler(event):
    """
    RunPod handler function.

    Input (event["input"]):
        - file: Base64 encoded PDF or image file
        - file_name: Optional filename for logging

    Output:
        - markdown: Extracted markdown text
        - page_count: Number of pages processed
        - success: Boolean indicating success
        - error: Error message if failed
    """
    try:
        job_input = event.get("input", {})

        # Get base64 encoded file
        file_data = job_input.get("file")
        if not file_data:
            return {
                "success": False,
                "error": "No file provided. Send base64 encoded PDF/image in 'file' field.",
            }

        file_name = job_input.get("file_name", "document.pdf")
        file_ext = Path(file_name).suffix.lower()

        # Decode base64 to bytes
        try:
            file_bytes = base64.b64decode(file_data)
        except Exception as e:
            return {
                "success": False,
                "error": f"Invalid base64 encoding: {e}",
            }

        log.info(f"Processing: {file_name} ({len(file_bytes)} bytes)")

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            all_markdown = []
            page_count = 0

            if file_ext == ".pdf":
                # Convert PDF to images
                log.info("Converting PDF to images...")
                images = convert_from_path(tmp_path, dpi=150)
                page_count = len(images)
                log.info(f"PDF has {page_count} pages")

                # Process each page
                for i, image in enumerate(images):
                    log.info(f"Processing page {i + 1}/{page_count}...")

                    # Save image temporarily
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as img_tmp:
                        image.save(img_tmp.name, "PNG")
                        img_path = img_tmp.name

                    try:
                        page_md = process_image(img_path)
                        all_markdown.append(f"<!-- Page {i + 1} -->\n{page_md}")
                    finally:
                        os.unlink(img_path)

            else:
                # Direct image processing
                page_count = 1
                page_md = process_image(tmp_path)
                all_markdown.append(page_md)

            # Combine all pages
            markdown = "\n\n---\n\n".join(all_markdown)

            log.info(f"Completed: {file_name} - {len(markdown)} chars, {page_count} pages")

            return {
                "success": True,
                "markdown": str(markdown),
                "page_count": int(page_count),
                "file_name": str(file_name),
            }

        finally:
            # Clean up temp file
            os.unlink(tmp_path)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        log.error(f"Error: {error_msg}")
        log.error(traceback.format_exc())

        return {
            "success": False,
            "error": error_msg,
        }


# Start the RunPod serverless worker
runpod.serverless.start({"handler": handler})
