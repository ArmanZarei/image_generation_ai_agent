import argparse
import json
import torch
from openai import OpenAI
from tools import ImageGenerationToolsContainer
from utils import load_pipe, dynamic_tool_call


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
