from openai import OpenAI
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import base64
import json
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI"))


def get_image_metadata(contents: bytes) -> dict:
    """Send image bytes to OpenAI and return {description, keywords}."""
    img = Image.open(BytesIO(contents))
    img.thumbnail((1024, 1024))

    buffer = BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=85)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[{
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Describe this image in 2-3 sentences. "
                        "Then give exactly 15 keywords. "
                        'Return JSON like: {"description": "...", "keywords": ["...", "..."]}'
                    ),
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{b64}",
                },
            ],
        }],
    )

    try:
        return json.loads(response.output_text)
    except (json.JSONDecodeError, AttributeError):
        return {"description": str(response.output_text), "keywords": []}
