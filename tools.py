import os
import logging
import json
from diffusers.utils import load_image
import time
from PIL import Image
import cv2
import numpy as np
from diffusers import StableDiffusionXLControlNetPipeline


class ImageGenerationToolsContainer:
    def __init__(self, pipe: StableDiffusionXLControlNetPipeline):
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
