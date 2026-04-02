import base64
import requests
from pathlib import Path
from langchain_ollama import ChatOllama
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_IMAGE_MODEL = "llava-phi3:latest"

llm = ChatOllama(base_url=BASE_URL, model=OLLAMA_MODEL, temperature=0)


class State(TypedDict):
    image_path: str
    query: str
    vision_output: str
    analysis: str
    final_answer: str


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def vision_node(state: State) -> dict:
    """Call Ollama /api/generate directly — the only reliable way to send images."""
    image_b64 = _encode_image(state["image_path"])

    payload = {
        "model": OLLAMA_IMAGE_MODEL,
        "prompt": (
            "Describe this image in detail. "
            "List every object, animal, person, text, colours, and scene context you can see. "
            "Be specific and thorough."
        ),
        "images": [image_b64],
        "stream": False,
    }

    response = requests.post(f"{BASE_URL}/api/generate", json=payload, timeout=120)
    response.raise_for_status()

    vision_text = response.json().get("response", "").strip()
    if not vision_text:
        raise ValueError(
            "Vision model returned empty output. "
            "Try: ollama pull llava-phi3"
        )

    return {"vision_output": vision_text}


def research_node(state: State) -> dict:
    prompt = f"""You are an expert image analyst.

A vision model described the image as:
\"\"\"
{state['vision_output']}
\"\"\"

User question: {state['query']}

Answer ONLY based on the description above. Do NOT invent details not mentioned."""

    response = llm.invoke(prompt)
    return {"analysis": response.content}


def writer_node(state: State) -> dict:
    prompt = f"""Write a clear, direct final answer for the user.
Only use facts from the analysis. Do not add new information.

Analysis:
{state['analysis']}"""

    response = llm.invoke(prompt)
    return {"final_answer": response.content}


builder = StateGraph(State)
builder.add_node("vision", vision_node)
builder.add_node("research", research_node)
builder.add_node("writer", writer_node)

builder.add_edge(START, "vision")
builder.add_edge("vision", "research")
builder.add_edge("research", "writer")
builder.add_edge("writer", END)

graph = builder.compile()


def run_agent(image_path: str, query: str) -> str:
    result = graph.invoke({"image_path": image_path, "query": query})
    return result["final_answer"]