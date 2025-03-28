import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import numpy as np
import json
import trimesh as Trimesh
from tqdm import tqdm

# Update imports for TripoSG
from .triposg.pipelines.pipeline_triposg import TripoSGPipeline
from .triposg.models.autoencoders import TripoSGVAEModel

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

import comfy.utils
import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar, common_upscale

script_directory = os.path.dirname(os.path.abspath(__file__))

# Create a simple logger since we don't have access to the Hunyuan utility
class Logger:
    def info(self, message):
        print(f"[TripoSG INFO] {message}")
    
    def error(self, message):
        print(f"[TripoSG ERROR] {message}")
        
    def warning(self, message):
        print(f"[TripoSG WARNING] {message}")

log = Logger()

class ComfyProgressCallback:
    def __init__(self, total_steps):
        self.pbar = ProgressBar(total_steps)
        
    def __call__(self, pipe, i, t, callback_kwargs):
        self.pbar.update(1)
        # Return only the keys that actually exist in callback_kwargs
        # Don't try to access keys that might not be there
        return callback_kwargs

class TripoSGModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": "", "tooltip": "Path to the TripoSG model either as a local path or 'pretrained' to download from HuggingFace"}),
            },
            "optional": {
                "attention_mode": (["sdpa", "sageattn"], {"default": "sdpa"}),
                "use_float16": ("BOOLEAN", {"default": True, "tooltip": "Use float16 precision for faster inference"}),
            }
        }

    RETURN_TYPES = ("TRIPOSG_MODEL",)
    RETURN_NAMES = ("triposg_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "TripoSG"
    DESCRIPTION = "Loads a TripoSG model for 3D mesh generation from a single image"

    def loadmodel(self, model_path, attention_mode="sdpa", use_float16=True):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        dtype = torch.float16 if use_float16 else torch.float32
        
        if model_path.lower() == "pretrained":
            # Download model from HuggingFace
            from huggingface_hub import snapshot_download
            triposg_weights_dir = os.path.join(script_directory, "pretrained_weights", "TripoSG")
            if not os.path.exists(triposg_weights_dir):
                log.info("Downloading TripoSG model from HuggingFace...")
                snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=triposg_weights_dir)
            model_path = triposg_weights_dir
        elif not os.path.exists(model_path):
            # Try to find in ComfyUI model paths
            possible_path = folder_paths.get_full_path("diffusion_models", model_path)
            if possible_path is not None:
                model_path = possible_path
            else:
                raise ValueError(f"Model path {model_path} does not exist")
        
        # Load TripoSG pipeline
        pipe = TripoSGPipeline.from_pretrained(model_path).to(device, dtype)
        
        return (pipe,)

class TripoSGImageToMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "triposg_model": ("TRIPOSG_MODEL",),
                "image": ("IMAGE",),
                "guidance_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 1000, "step": 1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "dense_octree_depth": ("INT", {"default": 8, "min": 4, "max": 12, "step": 1}),
                "hierarchical_octree_depth": ("INT", {"default": 9, "min": 5, "max": 12, "step": 1}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "TripoSG"
    DESCRIPTION = "Generates a 3D mesh from a single image using TripoSG"

    def process(self, triposg_model, image, guidance_scale, steps, seed, 
                dense_octree_depth, hierarchical_octree_depth):
        device = mm.get_torch_device()
        pipe = triposg_model
        
        # Convert ComfyUI image to PIL
        if image.shape[0] > 1:
            log.info("Multiple images detected, using only the first one")
        
        image_pil = Image.fromarray((image[0] * 255).numpy().astype(np.uint8))
        
        # Setup progress bar
        callback = ComfyProgressCallback(steps)
        
        # Generate mesh
        outputs = pipe(
            image=image_pil,
            generator=torch.Generator(device=pipe.device).manual_seed(seed),
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            dense_octree_depth=dense_octree_depth,
            hierarchical_octree_depth=hierarchical_octree_depth,
            callback_on_step_end=callback,
        ).samples[0]
        
        # Create trimesh
        mesh = Trimesh.Trimesh(outputs[0].astype(np.float32), np.ascontiguousarray(outputs[1]))
        
        return (mesh,)

class TripoSGMeshInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("TRIMESH", "INT", "INT", )
    RETURN_NAMES = ("trimesh", "vertices", "faces",)
    FUNCTION = "process"
    CATEGORY = "TripoSG"
    DESCRIPTION = "Display information about a 3D mesh"

    def process(self, trimesh):
        vertices = len(trimesh.vertices)
        faces = len(trimesh.faces)
        
        log.info(f"Mesh info: {vertices} vertices, {faces} faces")
        
        return (trimesh, vertices, faces)

class TripoSGExportMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "filename_prefix": ("STRING", {"default": "3D/TripoSG"}),
                "file_format": (["glb", "obj", "ply", "stl", "3mf", "dae"],),
            },
            "optional": {
                "save_file": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "process"
    CATEGORY = "TripoSG"
    OUTPUT_NODE = True
    DESCRIPTION = "Export a 3D mesh to a file"

    def process(self, trimesh, filename_prefix, file_format, save_file=True):
        output_dir = folder_paths.get_output_directory()
        out_path = os.path.join(output_dir, filename_prefix)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # Add timestamp to filename to avoid overwriting
        import time
        filename = f"{out_path}_{int(time.time())}.{file_format}"
        
        if save_file:
            log.info(f"Saving mesh to {filename}")
            trimesh.export(filename)
            
        return (filename,)

# Node mapping dictionaries for export
NODE_CLASS_MAPPINGS = {
    "TripoSGModelLoader": TripoSGModelLoader,
    "TripoSGImageToMesh": TripoSGImageToMesh,
    "TripoSGMeshInfo": TripoSGMeshInfo,
    "TripoSGExportMesh": TripoSGExportMesh
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TripoSGModelLoader": "TripoSG Model Loader",
    "TripoSGImageToMesh": "TripoSG Image to Mesh",
    "TripoSGMeshInfo": "TripoSG Mesh Info",
    "TripoSGExportMesh": "TripoSG Export Mesh"
} 