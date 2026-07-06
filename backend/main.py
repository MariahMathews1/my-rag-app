# =========================================================================
# main.py
#
# This is a basic RAG (Retrieval-Augmented Generation) backend using FastAPI.
#
# What it does, in order, every time the server starts up:
#   1. Reads all .txt files from the docs/ folder
#   2. Splits each file into small chunks
#   3. Sends each chunk to OpenAI to get its "embedding" (a list of numbers
#      representing its meaning)
#   4. Stores all of this in memory (a Python list) - no database needed
#
# Then, every time a user asks a question through the API:
#   5. Embeds the question the same way
#   6. Compares it against every stored chunk using cosine similarity
#   7. Grabs the top 3 most relevant chunks
#   8. Stuffs those chunks + the question into a prompt
#   9. Sends that prompt to OpenAI's chat model and returns the answer
# =========================================================================

import os
import glob
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ---- Setup -----------------------------------------------------------

# load_dotenv() reads the .env file and makes its values available
# through os.environ. This is how we keep the real API key out of the code.
load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

app = FastAPI()

# CORS: allows the React frontend (running on a different port) to call
# this backend without the browser blocking the request.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for a real app, replace * with your frontend's URL
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Step 1 & 2: Load and chunk the documents -------------------------

def chunk_text(text, max_words=120):
    """
    Splits one big string of text into smaller pieces (chunks).

    Why chunk at all? Embedding models work better on focused, shorter
    pieces of text rather than one giant document - and it means our
    similarity search can point to the SPECIFIC paragraph that answers
    a question, instead of returning an entire file.

    This is a simple word-count-based splitter. It's not perfect (it can
    cut a sentence in half), but it's easy to understand and good enough
    for a first RAG project.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks


def load_and_chunk_documents(folder_path="docs"):
    """
    Reads every .txt file in the given folder and breaks each one into
    chunks using chunk_text(). Returns a simple list of chunk strings.
    """
    all_chunks = []

    # glob finds every file matching a pattern - here, every .txt file
    # in the docs folder.
    file_paths = glob.glob(os.path.join(folder_path, "*.txt"))

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    return all_chunks


# ---- Step 3 & 4: Embed each chunk and store it in memory --------------

def get_embedding(text):
    """
    Sends a piece of text to OpenAI's embedding model and returns the
    resulting vector (a list of numbers representing the text's meaning).
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",  # a small, inexpensive embedding model
        input=text,
    )
    return response.data[0].embedding


def build_index():
    """
    Runs once when the server starts. Loads all document chunks, embeds
    each one, and stores everything together in a list of dictionaries.

    Each entry looks like:
        {"text": "the chunk's actual text", "embedding": [0.123, -0.045, ...]}

    This list IS our "vector database" - just kept in memory rather than
    in an actual database software. That's fine for a small, first project.
    """
    print("Building index from documents...")
    chunks = load_and_chunk_documents()

    index = []
    for chunk in chunks:
        embedding = get_embedding(chunk)
        index.append({"text": chunk, "embedding": embedding})

    print(f"Index built with {len(index)} chunks.")
    return index


# This runs ONE TIME, when the server first starts - not on every request.
# That's important: embedding is a paid API call, so we don't want to
# redo it every time a user asks a question.
document_index = build_index()


# ---- Step 6: Cosine similarity math ------------------------------------

def cosine_similarity(vector_a, vector_b):
    """
    Measures how similar two vectors are, returning a number from -1 to 1.
    1 means "pointing in exactly the same direction" (very similar meaning),
    0 means "unrelated", -1 means "opposite meaning".

    The math: dot product of the two vectors, divided by the product of
    their magnitudes (lengths). numpy handles the heavy lifting here.
    """
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )


def find_top_chunks(question_embedding, top_n=3):
    """
    Compares the question's embedding against every chunk's embedding in
    our index, and returns the top_n most similar chunks (highest
    cosine similarity scores).
    """
    scored_chunks = []
    for entry in document_index:
        score = cosine_similarity(question_embedding, entry["embedding"])
        scored_chunks.append((score, entry["text"]))

    # Sort by score, highest first, and take the top N.
    scored_chunks.sort(key=lambda pair: pair[0], reverse=True)
    top_chunks = [text for score, text in scored_chunks[:top_n]]
    return top_chunks


# ---- The API endpoint ---------------------------------------------------

# Pydantic model: defines the exact shape of the JSON the frontend must
# send. FastAPI automatically validates incoming requests against this.
class ChatRequest(BaseModel):
    question: str


@app.post("/api/chat")
def chat(request: ChatRequest):
    user_question = request.question

    # Step 5: embed the user's question the same way we embedded the chunks.
    question_embedding = get_embedding(user_question)

    # Step 6 & 7: find the most relevant chunks from our documents.
    top_chunks = find_top_chunks(question_embedding, top_n=3)

    # Step 8: build a prompt that includes the retrieved context PLUS
    # the original question. This is the "augmented" part of RAG - we're
    # augmenting the question with relevant background info before asking
    # the LLM to answer.
    context_text = "\n\n".join(top_chunks)
    prompt = f"""Use the following context to answer the question.
If the answer isn't in the context, say you don't know.

Context:
{context_text}

Question: {user_question}
"""

    # Step 9: send the augmented prompt to OpenAI's chat model.
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers based on the provided context."},
            {"role": "user", "content": prompt},
        ],
    )

    answer = response.choices[0].message.content
    return {"answer": answer, "sources_used": top_chunks}


# =========================================================================
# STREAMING VERSION
#
# Everything above (retrieval, prompt building) stays exactly the same.
# The only thing that changes is HOW we get the answer back from OpenAI
# and HOW we send it to the frontend: piece by piece, instead of waiting
# for the whole thing.
#
# The protocol we use here is simple and hand-rolled for learning purposes:
#   1. First we send one line of JSON containing the retrieved sources,
#      followed by a separator line "\n---STREAM-START---\n"
#   2. Then we send the answer as a series of raw text pieces, as OpenAI
#      generates them
#
# The frontend will know to split on that separator to tell the two
# parts apart.
# =========================================================================

async def generate_streaming_response(user_question: str):
    """
    This is a generator function - notice the `yield` keyword instead of
    `return`. A normal function runs once and gives back one value.
    A generator can pause, hand back a piece of data, then resume where
    it left off. FastAPI's StreamingResponse calls this repeatedly,
    sending each yielded piece to the browser immediately rather than
    waiting for the whole function to finish.
    """
    # --- Retrieval happens once, same as before, and is NOT streamed ---
    question_embedding = get_embedding(user_question)
    top_chunks = find_top_chunks(question_embedding, top_n=3)
    context_text = "\n\n".join(top_chunks)

    prompt = f"""Use the following context to answer the question.
If the answer isn't in the context, say you don't know.

Context:
{context_text}

Question: {user_question}
"""

    # First, send the sources as one JSON line, then our separator.
    # The frontend will grab everything before the separator as metadata.
    sources_payload = json.dumps({"sources": top_chunks})
    yield sources_payload + "\n---STREAM-START---\n"

    # Now call OpenAI with stream=True. Instead of one response object,
    # this gives us an iterator - something we can loop over - where each
    # item is a small "delta" (a fragment of the answer as it's generated).
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers based on the provided context."},
            {"role": "user", "content": prompt},
        ],
        stream=True,  # <-- this one flag changes everything
    )

    for chunk in stream:
        # Each chunk may or may not contain new text - sometimes it's
        # just metadata (like the model name), so we check first.
        delta_text = chunk.choices[0].delta.content
        if delta_text:
            yield delta_text


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest):
    """
    This endpoint returns a StreamingResponse instead of a plain dict.
    StreamingResponse takes our generator function above and sends each
    yielded piece to the browser as soon as it's produced, rather than
    waiting for the whole response to be built first.
    """
    return StreamingResponse(
        generate_streaming_response(request.question),
        media_type="text/plain",
    )