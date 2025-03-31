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
- FlashVDM from Tencent is available, compatible with mc and dmc (requires pip install diso)

![workflow (5)](https://github.com/user-attachments/assets/727c93b4-7d96-4068-bc41-e13ff1c94695)
![Captura de tela 2025-03-31 091513](https://github.com/user-attachments/assets/228a5b9e-c4e7-4dc7-9639-4a40802a7c3c)


## Aknowledgements
https://github.com/VAST-AI-Research/TripoSG  
https://github.com/kijai/ComfyUI-Hunyuan3DWrapper  
https://github.com/Tencent/FlashVDM
