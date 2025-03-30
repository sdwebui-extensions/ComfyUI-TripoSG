import os
import torch
import numpy as np
import trimesh as Trimesh
import folder_paths
import comfy.model_management as mm
from PIL import Image
from comfy.utils import ProgressBar

# TripoSG model imports
from .triposg.pipelines.pipeline_triposg import TripoSGPipeline


script_directory = os.path.dirname(os.path.abspath(__file__))

# Simple logger for TripoSG operations
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

    RETURN_TYPES = ("TRIPOSG_MODEL", "TRIPOSG_VAE")
    RETURN_NAMES = ("triposg_model", "triposg_vae")
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
        
        pipe = TripoSGPipeline.from_pretrained(model_path).to(device, dtype)
        
        return (pipe, pipe.vae)

class TripoSGVAEDecoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENTS", {"default": None}),
                "triposg_vae": ("TRIPOSG_VAE",),
                "bound_value": ("FLOAT", {"default": 1.005, "min": 0.5, "max": 2.0, "step": 0.005, "tooltip": "Single value for bounds (creates -value to +value for all dimensions)"}),
                "dense_octree_depth": ("INT", {"default": 8, "min": 4, "max": 12, "step": 1}),
                "hierarchical_octree_depth": ("INT", {"default": 9, "min": 5, "max": 12, "step": 1}),
                "chunk_size": ("INT", {"default": 256000, "min": 8192, "max": 512000, "step": 8192, "tooltip": "Number of points to process at once (higher values use more VRAM)"}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "TripoSG"
    DESCRIPTION = "Decodes latents to a 3D mesh using the TripoSG VAE decoder"

    def process(self, triposg_vae, latents, bound_value, dense_octree_depth, hierarchical_octree_depth, chunk_size):
        device = mm.get_torch_device()
        
        # Ensure latents are properly formatted
        if latents is not None:
            latents = latents.to(device=device, dtype=triposg_vae.dtype)
        else:
            log.error("No latents provided to VAE decoder")
            return (Trimesh.Trimesh(),)
        
        # Create symmetric bounds from the single value
        bounds = (-bound_value, -bound_value, -bound_value, bound_value, bound_value, bound_value)
        log.info(f"Using bounds: {bounds}")
        
        # Create geometric function for mesh extraction with memory-efficient chunking
        def geometric_func(x):
            num_points = x.shape[1]
            chunks = []
            
            total_chunks = (num_points + chunk_size - 1) // chunk_size
            pbar = ProgressBar(total_chunks)
            log.info(f"Processing {num_points} points in chunks of {chunk_size}")
            
            for i in range(0, num_points, chunk_size):
                chunk = x[:, i:i+chunk_size, :]
                chunk_result = triposg_vae.decode(latents, sampled_points=chunk).sample
                chunks.append(chunk_result)
                pbar.update(1)
                
            return torch.cat(chunks, dim=1)
        
        from .triposg.inference_utils import hierarchical_extract_geometry
        
        # Memory usage warning for high octree depth values
        total_depth = dense_octree_depth + hierarchical_octree_depth
        if total_depth > 18:
            log.warning(f"High octree depth values (total: {total_depth}) may cause memory issues")
        
        try:
            output = hierarchical_extract_geometry(
                geometric_func,
                device,
                bounds=bounds,
                dense_octree_depth=dense_octree_depth,
                hierarchical_octree_depth=hierarchical_octree_depth,
            )
            
            if output is None or len(output) == 0:
                log.error("Mesh extraction failed, likely due to memory constraints with high octree depths")
                return (Trimesh.Trimesh(),)
                
            # Create trimesh from extracted geometry
            mesh = Trimesh.Trimesh(output[0][0].astype(np.float32), np.ascontiguousarray(output[0][1]))
            
            return (mesh,)
        except Exception as e:
            log.error(f"Error during mesh extraction: {str(e)}")
            log.error("Try reducing octree depth values or increasing chunk size")
            return (Trimesh.Trimesh(),)

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
            }
        }

    RETURN_TYPES = ("LATENTS",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "process"
    CATEGORY = "TripoSG"
    DESCRIPTION = "Generates latents from a single image using TripoSG (without decoding to mesh)"

    def process(self, triposg_model, image, guidance_scale, steps, seed):
        device = mm.get_torch_device()
        pipe = triposg_model
        
        # Convert ComfyUI image to PIL
        if image.shape[0] > 1:
            log.info("Multiple images detected, using only the first one")
        
        image_pil = Image.fromarray((image[0] * 255).numpy().astype(np.uint8))
        
        # Setup progress tracking
        callback = ComfyProgressCallback(steps)
        
        # Prepare latents with proper seed for reproducibility
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        
        # Encode input image
        image_embeds, negative_image_embeds = pipe.encode_image(
            image_pil, device=pipe.device, num_images_per_prompt=1
        )
        
        # Set up generation parameters
        pipe._num_timesteps = steps
        pipe._guidance_scale = guidance_scale
        pipe.scheduler.set_timesteps(steps, device=pipe.device)
        timesteps = pipe.scheduler.timesteps
        
        # Initialize latent space representation
        latents_dtype = image_embeds.dtype
        hidden_size = 64  # Transformer input dimension
        sample_size = 2048  # Standard sample size for TripoSG
        latents_shape = (1, sample_size, hidden_size)
        latents = torch.randn(latents_shape, generator=generator, device=pipe.device, dtype=latents_dtype)
        
        # Run denoising diffusion process
        for i, t in enumerate(timesteps):
            # Handle classifier-free guidance if enabled
            latent_model_input = torch.cat([latents, latents], dim=0) if pipe.do_classifier_free_guidance else latents
            
            # Create appropriate timestep tensor
            if pipe.do_classifier_free_guidance:
                timestep = torch.tensor([t, t], device=pipe.device)
            else:
                timestep = torch.tensor([t], device=pipe.device)
            
            # Get model prediction
            noise_pred = pipe.transformer(
                latent_model_input,
                timestep=timestep,
                encoder_hidden_states=torch.cat([negative_image_embeds, image_embeds]) if pipe.do_classifier_free_guidance else image_embeds,
                return_dict=False,
            )[0]
            
            # Apply classifier-free guidance
            if pipe.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Update latents with scheduler step
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
            # Update progress
            if callback is not None:
                callback_kwargs = {"latents": latents}
                callback(pipe, i, t, callback_kwargs)
                
        return (latents,)

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
        
        # Create directory if needed
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # Add timestamp to prevent overwriting
        import time
        filename = f"{out_path}_{int(time.time())}.{file_format}"
        
        if save_file:
            log.info(f"Saving mesh to {filename}")
            trimesh.export(filename)
            
        return (filename,)

# Node mapping for ComfyUI integration
NODE_CLASS_MAPPINGS = {
    "TripoSGModelLoader": TripoSGModelLoader,
    "TripoSGImageToMesh": TripoSGImageToMesh,
    "TripoSGVAEDecoder": TripoSGVAEDecoder,
    "TripoSGMeshInfo": TripoSGMeshInfo,
    "TripoSGExportMesh": TripoSGExportMesh
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TripoSGModelLoader": "TripoSG Model Loader",
    "TripoSGImageToMesh": "TripoSG Image to Latents",
    "TripoSGVAEDecoder": "TripoSG VAE Decoder",
    "TripoSGMeshInfo": "TripoSG Mesh Info",
    "TripoSGExportMesh": "TripoSG Export Mesh"
} 