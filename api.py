from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from image_processor import get_image_metadata
import json
import os
import numpy as np


model = SentenceTransformer("all-MiniLM-L6-v2")

with open("image_results.json") as f:
    data = json.loads(f.read())

texts = [row["result"]["description"] for row in data]
embeds = model.encode(texts, normalize_embeddings=True)

app = FastAPI()

app.mount("/images", StaticFiles(directory="images"), name="images")


@app.get("/")
async def serve_ui():
    return FileResponse("index.html")


@app.get("/search")
async def search(query: str = ""):
    if not query.strip():
        return [
            {"filename": row["filename"], "description": row["result"]["description"], "score": 1.0}
            for row in data
        ]

    q_embed = model.encode(query, normalize_embeddings=True)
    scores = embeds @ q_embed
    indices = np.argsort(scores)[::-1]

    return [
        {
            "filename": data[i]["filename"],
            "description": data[i]["result"]["description"],
            "score": float(scores[i]),
        }
        for i in indices
    ]


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global data, texts, embeds

    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    contents = await file.read()

    # Get description + keywords via OpenAI
    metadata = get_image_metadata(contents)

    # Resolve filename conflicts
    filename = file.filename
    save_path = os.path.join("images", filename)
    if os.path.exists(save_path):
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{len(data)}{ext}"
        save_path = os.path.join("images", filename)

    with open(save_path, "wb") as f:
        f.write(contents)

    # Update in-memory state
    new_entry = {"filename": filename, "result": metadata}
    data.append(new_entry)
    texts.append(metadata["description"])
    new_embed = model.encode(metadata["description"], normalize_embeddings=True)
    embeds = np.vstack([embeds, new_embed.reshape(1, -1)])

    # Persist
    with open("image_results.json", "w") as f:
        json.dump(data, f, indent=2)

    return {"filename": filename, "result": metadata}
