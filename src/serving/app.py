from contextlib import asynccontextmanager
import os

from dotenv import load_dotenv
from fastapi import FastAPI, status
from langchain_core.language_models import BaseChatModel

from src.data.factory import make_embeddings, make_model, make_vector_store
from src.data.retrieval import retrieve_from_query
from src.serving.chain import create_prompt
from src.serving.schemas import AskRequest

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
async def ask(query: AskRequest):

    data = retrieve_from_query(artifacts['storage'], query.question, 10)
    prompt = create_prompt(data, query.question)

    model: BaseChatModel = artifacts['model']
    response = await model.ainvoke(prompt)

    return {'response': response.text}
