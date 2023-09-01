from fastapi import FastAPI, Depends, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import Request
import typing as t
import uvicorn
import os
from llm_engine import FaissIndex
import openai


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

app = FastAPI(
    title = "Portfolio LLM Backend",
    description = "Backend for Portfolio LLM",
    docs_url  = "/docs",
)

origins = [
    "https://ashwinrachha.onrender.com",
    "http://localhost:8000",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

faiss_index = FaissIndex()

class UserQuery(BaseModel):
    query: str

@app.get("/")
async def root(request: Request):
    return {"Message" : "Server is Up and Running"}

@app.post("/query")
async def query(user_query: UserQuery):
    query = user_query.query
    response = faiss_index.qa_chain({"query" : query})
    return {"response" : response["result"]}

if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port = 8000)