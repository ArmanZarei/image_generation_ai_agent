# How to run

Create a python env. and install required libraries
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

then
```
python main.py ----openai_api_key="OPENAI_API_KEY"
```

# Example

${\text{\color{orange}Assistant:}}$
I'm here to help you to generate images! You can either describe an image you have in mind and I can generated it for you, or use a layout of an already generated image to generate a new version of it.
Type 'exit' or 'quit' to stop the conversation.
Let's start!

${\text{\color{gray}User:}}$ Hi, what capabilities do you have?

${\text{\color{orange}Assistant:}}$ I can assist you with various tasks related to image generation and manipulation, including:
1. **Generate Images from Text Prompts**: I can create images based on descriptive text prompts.
2. **Extract Canny Edges from Images**: I can take an existing image and extract its outlines or edges.
3. **Generate Images from Canny Edges**: Using a Canny edge image and a text prompt, I can create a new image that incorporates the outlines and follows the described content.

These capabilities allow for creative exploration and modification of visual content. If you have a specific task or idea in mind, feel free to let me know!

${\text{\color{gray}User:}}$ Cool! Can you generate an image of a cat with sunglasses?

${\text{\color{orange}Assistant:}}$ I have generated an image of a cat wearing sunglasses. You can view it [here](images/a cat wearing sunglasses_1744047920.350298.png).

![a cat wearing sunglasses](./images/a%20cat%20wearing%20sunglasses_1744047920.350298.png)

${\text{\color{gray}User:}}$ Nice! How about an image of an elephant with a hat?

${\text{\color{orange}Assistant:}}$ I have created an image of an elephant wearing a hat. You can view it [here](images/an elephant wearing a hat_1744047954.8751402.png).

![an elephant wearing a hat](images/an%20elephant%20wearing%20a%20hat_1744047954.8751402.png)

${\text{\color{gray}User:}}$ Awesome! Can you generate the layout (canny edges) of the previous generated image (the cat one)?

${\text{\color{orange}Assistant:}}$ I have extracted the canny edges (layout) of the image of a cat wearing sunglasses. You can view it [here](images/a cat wearing sunglasses_1744047920_canny.png).

![a cat wearing sunglasses](images/a%20cat%20wearing%20sunglasses_1744047920_canny.png)

${\text{\color{gray}User:}}$ Can you use the layout and generate an image of a dog?

${\text{\color{orange}Assistant:}}$ I have generated an image of a dog using the layout from the cat image. You can view it [here](images/a dog_controlnet_1744048021.6299403.png).

![a dog](images/a%20dog_controlnet_1744048021.6299403.png)

${\text{\color{gray}User:}}$ exit
