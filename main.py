import argparse
import json
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
import cv2
from PIL import Image
import time
from openai import OpenAI
import os
import logging


class ImageGenerationToolsContainer:
    def __init__(self, pipe):
        self.pipe = pipe

        if not os.path.exists("images"):
            os.makedirs("images")

        self.logger = self.__set_up_logger()
    
    def __set_up_logger(self):
        logger = logging.getLogger("ImageGenerationToolsContainer")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler("tools.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger
    
    def get_tools_desc(self):
        with open("tools.json", "r") as f:
            tools = json.load(f)
        return tools

    def image_generation_tool(self, prompt: str) -> str:
        self.logger.info(f"[image_generation_tool] Generating image with prompt: \"{prompt}\"")

        save_path = f"images/{prompt}_{time.time()}.png"
        self.pipe(
            prompt,
            image=Image.fromarray(np.zeros((1024, 1024))),
            controlnet_conditioning_scale=0.,
        ).images[0].save(save_path)

        return save_path

    def extract_canny_edges_tool(self, image_path: str) -> str:
        self.logger.info(f"[extract_canny_edges_tool] Extracting Canny edges from image: \"{image_path}\"")

        orig_image = load_image(image_path)
        image = np.array(orig_image)

        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        
        save_path = f"{image_path.rsplit('.')[0]}_canny.png"
        Image.fromarray(image).save(save_path)
        
        return save_path

    def controlnet_image_generation_tool(self, prompt: str, canny_image_path: str) -> str:
        self.logger.info(f"[controlnet_image_generation_tool] Generating image with ControlNet using prompt: \"{prompt}\" and Canny image: \"{canny_image_path}\"")

        save_path = f"images/{prompt}_controlnet_{time.time()}.png"

        self.pipe(
            prompt,
            negative_prompt='low quality, bad quality, sketches',
            image=load_image(canny_image_path),
            controlnet_conditioning_scale=0.5,
        ).images[0].save(save_path)

        return save_path


def load_pipe():
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


def dynamic_tool_call(image_generation_tools_container, method_name, arguments):
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


def parse_args():
    parser = argparse.ArgumentParser(description="Image Generation with AI Agent")
    
    parser.add_argument("--openai_api_key", type=str, required=True, help="OpenAI API key")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    image_generation_tools_container = ImageGenerationToolsContainer(pipe=load_pipe())

    client = OpenAI(api_key=args.openai_api_key)
    tools = image_generation_tools_container.get_tools_desc()
    messages = [{"role": "system", "content": "You are a helpful assistant that can generate images."}]
    

    print("*"*100)
    print("I'm here to help you to generate images! You can either describe an image you have in mind and I can generated it for you, or use a layout of an already generated image to generate a new version of it.\nType 'exit' or 'quit' to stop the conversation.\nLet's start!")
    print("*"*100)
    while True:
        user_input = input("User: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            break

        messages.append({"role": "user", "content": user_input})

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )

        response = completion.choices[0].message

        # If the model wants to call a tool
        if response.tool_calls:
            for tool_call in response.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                messages.append(response)

                tool_response = dynamic_tool_call(image_generation_tools_container, function_name, arguments)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_response,
                })

            # Re-send the conversation after the tool call for a follow-up response if needed
            follow_up_completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
            )
            follow_up_message = follow_up_completion.choices[0].message
            messages.append(follow_up_message)
            print(f"Assistant: {follow_up_message.content}")

        else:
            messages.append(response)
            print(f"Assistant: {response.content}")
