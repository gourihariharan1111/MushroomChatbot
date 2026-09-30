
import os
import base64

import gradio as gr
from google import genai

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

SYSTEM_INSTRUCTION = """
You are a mushroom expert. You have good knowledge of mushrooms
and can identify them.

Your conversation should stay focused on mushrooms, fungi,
mushroom identification, mushroom biology, ecology, appearance,
edibility and safety.

If the user asks about any unrelated topic, politely come back
to discussion on mushrooms.

You must never state that a mushroom is safe to eat based only
on an image.

Mushroom identification from photographs can be uncertain.
If you are uncertain about edibility of a mushroom, communicate
the uncertainty and recommend verification by a qualified local
expert before consumption.

If you get a question on toxic mushrooms such as Amanita muscaria,
alert the user that there is a potential risk if consumed.
"""

default_model = "gemini-3.8-flash"

def input2output_chatbot(inputs, history, request: gr.Request):

    model_inputs = []
    if inputs.get("text"):

        model_inputs.append({
            "type": "text",
            "text": inputs["text"]
        })

    if "files" in inputs:

        for file_path in inputs["files"]:

            print("IMAGE FILE:", file_path)

            with open(file_path, "rb") as f:
                image_bytes = f.read()

            # Convert binary image to base64
            image_b64 = base64.b64encode(
                image_bytes
            ).decode("utf-8")

            model_inputs.append({
                "type": "image",
                "data": image_b64,
                "mime_type": "image/jpeg"
            })

    print("Model inputs:", model_inputs)

    response = client.interactions.create(
        model=default_model,
        input=model_inputs,
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config={
            "thinking_level": "low"
        },
        store=False
    )
    print("Gemini responded!")

    answer = response.output_text
    print(answer)
    return answer


if "demo" in locals() and demo.is_running:
    demo.close()
demo = gr.ChatInterface(
    input2output_chatbot,
    multimodal=True
)
demo.launch(
    server_name="0.0.0.0",
    server_port=7860
)