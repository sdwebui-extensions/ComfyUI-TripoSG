import os
import torch
import numpy as np
import trimesh as Trimesh
import folder_paths
import comfy.model_management as mm
from PIL import Image
from comfy.utils import ProgressBar
import torch.nn as nn

# TripoSG model imports
from .triposg.pipelines.pipeline_triposg import TripoSGPipeline
# FlashVDM imports
from .FlashVDM import FlashVDMCrossAttentionProcessor, FlashVDMTopMCrossAttentionProcessor
from .FlashVDM.geometry import extract_near_surface_points, get_neighbor
from .FlashVDM.point_processing import process_grid_points, reshape_grid_logits, group_points_for_processing


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
        #log.info(f"Using bounds: {bounds}")
        
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

class TripoSGFlashVDMDecoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENTS", {"default": None}),
                "triposg_vae": ("TRIPOSG_VAE",),
                "bound_value": ("FLOAT", {"default": 1.005, "min": 0.5, "max": 2.0, "step": 0.005, "tooltip": "Single value for bounds (creates -value to +value for all dimensions)"}),
                "dense_octree_depth": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 64}),
                "chunk_size": ("INT", {"default": 128000, "min": 1000, "max": 512000, "step": 1000, "tooltip": "Number of points to process at once (higher values use more VRAM)"}),
                "mc_algo": (["mc", "dmc"], {"default": "mc", "tooltip": "Marching cubes algorithm: standard (MC) or Direct Marching Cubes (DMC) from Diso"}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "TripoSG"
    DESCRIPTION = "Decodes latents to a 3D mesh using the TripoSG VAE decoder with FlashVDM for faster performance"

    def process(self, triposg_vae, latents, bound_value, dense_octree_depth, chunk_size, mc_algo="mc"):
        mini_grid_num = 1
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
        
        # Create FlashVDM processor based on the mode
        processor = FlashVDMCrossAttentionProcessor()
        
        # Apply FlashVDM to the VAE decoder
        # Since we don't have direct access to change the attention processor in TripoSG,
        # we'll implement a similar approach by creating a FlashVDM-enabled geometric function
            
        # First, we'll extract all query points using the hierarchical approach
        from .triposg.inference_utils import generate_grid_points
        
        # 1. Generate query points
        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min
        
        octree_resolution = dense_octree_depth
        min_resolution = 63
        
        resolutions = []
        if octree_resolution < min_resolution:
            resolutions.append(octree_resolution)
        while octree_resolution >= min_resolution:
            resolutions.append(octree_resolution)
            octree_resolution = octree_resolution // 2
        resolutions.reverse()
        resolutions[0] = round(resolutions[0] / mini_grid_num) * mini_grid_num - 1
        for i, resolution in enumerate(resolutions[1:]):
            resolutions[i + 1] = resolutions[0] * 2 ** (i + 1)
            
        log.info(f"FlashVDM Resolution: {resolutions}")
        
        # Generate initial grid
        xyz_samples, grid_size, _ = generate_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=resolutions[0],
            indexing="ij"
        )
        
        # Process with FlashVDM approach
        dilate = nn.Conv3d(1, 1, 3, padding=1, bias=False, device=device, dtype=triposg_vae.dtype)
        dilate.weight = torch.nn.Parameter(torch.ones(dilate.weight.shape, dtype=triposg_vae.dtype, device=device))
        
        grid_size = np.array(grid_size)
        
        # Process points using FlashVDM point processing utilities
        xyz_samples, mini_grid_size = process_grid_points(
            xyz_samples=xyz_samples, 
            device=device, 
            dtype=triposg_vae.dtype, 
            batch_size=latents.shape[0], 
            mini_grid_num=mini_grid_num
        )
        
        # Decode using batched processing
        batch_logits = []
        num_batches = max(chunk_size // xyz_samples.shape[1], 1)
        pbar = ProgressBar(xyz_samples.shape[0])
        log.info(f"Processing {xyz_samples.shape[0]} batches with {xyz_samples.shape[1]} points each")
        
        processor.topk = True
        
        for start in range(0, xyz_samples.shape[0], num_batches):
            queries = xyz_samples[start: start + num_batches, :]
            batch = queries.shape[0]
            
            # Repeat latents for batch processing
            batch_latents = torch.repeat_interleave(latents, batch, dim=0)
            
            # Decode queries
            logits = triposg_vae.decode(batch_latents, sampled_points=queries).sample
            batch_logits.append(logits)
            pbar.update(1)
            
        # Reshape the results to grid
        grid_logits = reshape_grid_logits(
            batch_logits=batch_logits,
            batch_size=latents.shape[0],
            grid_size=grid_size,
            mini_grid_num=mini_grid_num,
            mini_grid_size=mini_grid_size
        )
        
        # Process hierarchical octree levels
        mc_level = 0.0  # Default marching cubes threshold
        for octree_depth_now in resolutions[1:]:
            grid_size = np.array([octree_depth_now + 1] * 3)
            resolution = bbox_size / octree_depth_now
            next_index = torch.zeros(tuple(grid_size), dtype=triposg_vae.dtype, device=device)
            next_logits = torch.full(next_index.shape, -10000., dtype=triposg_vae.dtype, device=device)
            
            # Find near-surface points
            curr_points = extract_near_surface_points(grid_logits.squeeze(0), mc_level)
            curr_points += grid_logits.squeeze(0).abs() < 0.95
            
            # Determine expansion amount
            if octree_depth_now == resolutions[-1]:
                expand_num = 0
            else:
                expand_num = 1
                
            # Dilate points if needed
            for i in range(expand_num):
                curr_points = dilate(curr_points.unsqueeze(0).to(triposg_vae.dtype)).squeeze(0)
                
            # Get indices of points to evaluate
            (cidx_x, cidx_y, cidx_z) = torch.where(curr_points > 0)
            
            # Set up next level indices
            next_index[cidx_x * 2, cidx_y * 2, cidx_z * 2] = 1
            for i in range(2 - expand_num):
                next_index = dilate(next_index.unsqueeze(0)).squeeze(0)
            nidx = torch.where(next_index > 0)
            
            # Convert indices to 3D points
            next_points = torch.stack(nidx, dim=1)
            next_points = (next_points * torch.tensor(resolution, dtype=torch.float32, device=device) +
                        torch.tensor(bbox_min, dtype=torch.float32, device=device))
            
            # Group points for efficient processing
            query_grid_num = 6
            next_points, index, unique_values = group_points_for_processing(next_points, query_grid_num)
            
            # Initialize grid for current level
            grid_logits_current = torch.zeros((next_points.shape[1]), dtype=latents.dtype, device=latents.device)
            
            # Process points in chunks
            input_grid = [[], []]
            logits_grid_list = []
            start_num = 0
            sum_num = 0
            
            # Group points by grid cell for more efficient processing
            for grid_index, count in zip(unique_values[0].cpu().tolist(), unique_values[1].cpu().tolist()):
                if sum_num + count < chunk_size or sum_num == 0:
                    sum_num += count
                    input_grid[0].append(grid_index)
                    input_grid[1].append(count)
                else:
                    # Process current batch
                    processor.topk = input_grid
                    queries = next_points[:, start_num:start_num + sum_num]
                    logits_grid = triposg_vae.decode(latents, sampled_points=queries).sample
                    start_num = start_num + sum_num
                    logits_grid_list.append(logits_grid)
                    
                    # Start new batch
                    input_grid = [[grid_index], [count]]
                    sum_num = count
                    
            # Process remaining points
            if sum_num > 0:
                processor.topk = input_grid
                queries = next_points[:, start_num:start_num + sum_num]
                logits_grid = triposg_vae.decode(latents, sampled_points=queries).sample
                logits_grid_list.append(logits_grid)
                
            # Combine results
            logits_grid = torch.cat(logits_grid_list, dim=1)
            grid_logits_current[index.indices] = logits_grid.squeeze(0).squeeze(-1)
            next_logits[nidx] = grid_logits_current
            grid_logits = next_logits.unsqueeze(0)
            
        # Replace invalid values with NaN
        grid_logits[grid_logits == -10000.] = float('nan')
        
        # Convert final grid to mesh using marching cubes or DMC
        if mc_algo == "mc":
            try:
                from skimage import measure
                
                # Extract mesh using marching cubes
                volume = grid_logits.squeeze().detach().cpu().numpy()
                spacing = (bbox_max - bbox_min) / np.array(volume.shape)
                
                # Run marching cubes
                verts, faces, normals, values = measure.marching_cubes(
                    volume, 
                    level=0.0,  # Isosurface level
                    spacing=spacing,
                    method='lewiner'
                )
                
                # Adjust vertices position
                verts += bbox_min
                
                # Create trimesh
                mesh = Trimesh.Trimesh(verts, faces)
                
                return (mesh,)
            except Exception as e:
                log.error(f"Error during mesh extraction: {str(e)}")
                log.error("Try reducing octree depth values or increasing chunk size")
                return (Trimesh.Trimesh(),)
        else:  # DMC from Diso
            try:
                # Try to import DiSo library for DMC
                try:
                    from diso import DiffDMC
                except ImportError:
                    log.warning("DiSo not found, installing it...")
                    import subprocess
                    # Install DiSo for DMC support
                    subprocess.check_call(["pip", "install", "diso"])
                    from diso import DiffDMC
                
                # Extract mesh using DMC
                volume = grid_logits.squeeze().detach().cpu().numpy()
                
                # Create DMC extractor
                dmc = DiffDMC(dtype=torch.float32).to(device)
                
                # Convert volume to SDF format expected by DMC - ensure it's on the device
                sdf = -torch.from_numpy(volume).to(device, dtype=torch.float32) / dense_octree_depth
                sdf = sdf.contiguous()
                
                # Convert bbox_min to tensor on the right device
                bbox_min_tensor = torch.tensor(bbox_min, dtype=torch.float32, device=device)
                bbox_max_tensor = torch.tensor(bbox_max, dtype=torch.float32, device=device)
                
                # Run DMC
                verts, faces = dmc(sdf, deform=None, return_quads=False, normalize=True)
                
                # Center vertices and convert to numpy
                verts = verts - 0.5
                verts = verts * (bbox_max_tensor - bbox_min_tensor)
                verts = verts + bbox_min_tensor
                
                # Move to CPU before converting to numpy
                vertices = verts.detach().cpu().numpy()
                faces = faces.detach().cpu().numpy()[:, ::-1]  # Reverse face winding order
                
                # Create trimesh
                mesh = Trimesh.Trimesh(vertices, faces)
                
                return (mesh,)
            except Exception as e:
                log.error(f"Error during DMC mesh extraction: {str(e)}")
                log.error("Try reducing octree depth values or increasing chunk size")
                return (Trimesh.Trimesh(),)

# Node mapping for ComfyUI integration
NODE_CLASS_MAPPINGS = {
    "TripoSGModelLoader": TripoSGModelLoader,
    "TripoSGImageToMesh": TripoSGImageToMesh,
    "TripoSGVAEDecoder": TripoSGVAEDecoder,
    "TripoSGMeshInfo": TripoSGMeshInfo,
    "TripoSGExportMesh": TripoSGExportMesh,
    "TripoSGFlashVDMDecoder": TripoSGFlashVDMDecoder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TripoSGModelLoader": "TripoSG Model Loader",
    "TripoSGImageToMesh": "TripoSG Image to Latents",
    "TripoSGVAEDecoder": "TripoSG VAE Decoder",
    "TripoSGMeshInfo": "TripoSG Mesh Info",
    "TripoSGExportMesh": "TripoSG Export Mesh",
    "TripoSGFlashVDMDecoder": "TripoSG FlashVDM Decoder"
} 