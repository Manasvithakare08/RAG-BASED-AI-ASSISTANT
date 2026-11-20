import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests  # type: ignore


# ---------------- EMBEDDING FUNCTION ----------------
def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    data = r.json()

    
    if "embedding" in data:
        return [data["embedding"]]
    elif "embeddings" in data:
        return data["embeddings"]
    else:
        raise ValueError(f"Unexpected response from Ollama: {data}")

# ---------------- LLM INFERENCE ----------------
def inference(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )

    data = r.json()


    if "response" in data:
        return data["response"]
    elif "error" in data:
        raise ValueError(f"Ollama Error: {data['error']}")
    else:
        raise ValueError(f"Unexpected response from Ollama: {data}")


# ---------------- MAIN QUERY FUNCTION ----------------
def process_query(incoming_query, top_k=5):

    df = joblib.load("embeddings.joblib")

    question_embedding = create_embedding([incoming_query])[0]

    
    embedding_column = "embedding"  

    similarities = cosine_similarity(
        np.vstack(df[embedding_column]),
        [question_embedding]
    ).flatten()

    max_indx = similarities.argsort()[::-1][:top_k]
    new_df = df.loc[max_indx]

    prompt = f"""
I am teaching web development in my Sigma web development course. 
Here are video subtitle chunks:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}

User Question:
"{incoming_query}"

Answer in a human way.
Mention video number and timestamp.
"""

    response = inference(prompt)

    return response, new_df 