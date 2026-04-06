import os
import base64
from langchain_core.messages import HumanMessage
import streamlit as st

try:
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:
    pass

import requests
from pathlib import Path
from langchain_ollama import ChatOllama
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

LLM_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

llm = ChatGroq(model=LLM_MODEL, api_key=os.getenv("GROQ_API_KEY"), temperature=0.7)
vision_llm = ChatGroq(model=VISION_MODEL, api_key=os.getenv("GROQ_API_KEY"), temperature=0.3)

class State(TypedDict):
    image_path: str
    query: str
    vision_output: str
    analysis: str
    final_answer: str


def _encode_image(image_path: str) -> tuple[str, str]:
    """Returns (base64_data, media_type)."""
    ext = Path(image_path).suffix.lower().lstrip(".")
    media_type_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                      "png": "image/png", "webp": "image/webp", "gif": "image/gif"}
    media_type = media_type_map.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), media_type



def vision_node(state: State) -> dict:
    image_b64, media_type = _encode_image(state["image_path"])

    message = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{image_b64}"
            },
        },
        {
            "type": "text",
            "text": (
                "Describe this image in detail. "
                "List every object, animal, person, text, colours, and scene context you can see. "
                "Be specific and thorough."
            ),
        },
    ])

    response = vision_llm.invoke([message])
    vision_text = response.content.strip()
    if not vision_text:
        raise ValueError("Vision model returned empty output.")
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