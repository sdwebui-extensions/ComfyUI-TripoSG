# TripoSG Nodes for ComfyUI
Created by Alfredo Fernandes inspired by Hunyuan3D nodes by Kijai

This extension adds TripoSG 3D mesh generation capabilities to ComfyUI, allowing you to generate 3D meshes from a single image using the TripoSG model.

## Installation

1. Clone the model to "comfyui/models/diffusers":
  ```
  git clone https://huggingface.co/VAST-AI/TripoSG
  ```
1. Clone the repository to your custom_nodes folder:
  ```
  git clone https://github.com/fredconex/ComfyUI-TripoSG.git
  cd ComfyUI-TripoSG
  ```
2. Install dependencies(For portable use python embeded):
  ```
  pip install -r requirements.txt
  ```
Warning: Torch Cluster takes a while to build, that's expected!

## Notes

- Higher values for the octree depth parameters will result in more detailed meshes but require more VRAM and processing time.
- TripoSG model outputs a clean mesh that usually doesn't require post-processing.
- You can use the standard ComfyUI mesh viewing and processing nodes with the output from TripoSG.

![workflow (1)](https://github.com/user-attachments/assets/9785b543-ac26-4205-80bb-002d1b2f5993)

## TripoSG Repository
https://github.com/VAST-AI-Research/TripoSG
