from contextlib import asynccontextmanager
import os

from dotenv import load_dotenv
from fastapi import FastAPI, status
from langchain_core.language_models import BaseChatModel

from src.chain import create_prompt
from src.factory import make_embeddings, make_model, make_vector_store
from src.retrieval import retrieve_from_query

artifacts = {}


@asynccontextmanager
async def lifespan(a: FastAPI):
    load_dotenv()
    embedding = make_embeddings(device=os.getenv('DEVICE'))
    vector_store = make_vector_store(os.getenv('INDEX_NAME'), embedding)
    artifacts['storage'] = vector_store

    model: BaseChatModel = make_model()
    artifacts['model'] = model

    yield


app = FastAPI(lifespan=lifespan)


@app.post('/ask', status_code=status.HTTP_200_OK)
async def ask(query: str):

    data = retrieve_from_query(artifacts['storage'], query, 10)
    prompt = create_prompt(data, query)

    model: BaseChatModel = artifacts['model']
    response = model.invoke(prompt).text

    return {'response': response}
