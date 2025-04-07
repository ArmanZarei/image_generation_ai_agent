import torch
from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetPipeline
from tools import ImageGenerationToolsContainer


def load_pipe() -> StableDiffusionXLControlNetPipeline:
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")
    pipe.set_progress_bar_config(leave=False)

    return pipe


def dynamic_tool_call(image_generation_tools_container: ImageGenerationToolsContainer, method_name: str, arguments) -> str:
    if method_name == "image_generation_tool":
        image_path = image_generation_tools_container.image_generation_tool(**arguments)
        tool_response = f"Image successfully generated and saved to: {image_path}"
    elif method_name == "extract_canny_edges_tool":
        image_path = image_generation_tools_container.extract_canny_edges_tool(**arguments)
        tool_response = f"Canny edge image saved to: {image_path}"
    elif method_name == "controlnet_image_generation_tool":
        image_path = image_generation_tools_container.controlnet_image_generation_tool(**arguments)
        tool_response = f"Image generated using ControlNet and saved to: {image_path}"
    else:
        tool_response = "Sorry, I don't recognize that tool."
    
    return tool_response
