from contextlib import asynccontextmanager
import os

from dotenv import load_dotenv
from fastapi import FastAPI, status

from src.data.factory import make_embeddings, make_vector_store
from src.rag.factory import create_agent_tools, make_agent
from src.serving.schemas import AskRequest, AskResponse


@asynccontextmanager
async def lifespan(a: FastAPI):
    load_dotenv()
    embedding = make_embeddings(device=os.getenv('DEVICE'))
    vector_store = make_vector_store(os.getenv('INDEX_NAME'), embedding)
    agent_tools = create_agent_tools(vector_store)

    agent = make_agent(agent_tools)
    app.state.agent = agent

    yield


app = FastAPI(lifespan=lifespan)


@app.post('/ask', status_code=status.HTTP_200_OK)
async def ask(query: AskRequest):
    inputs = {'messages': [('user', query.question)]}
    answer = await app.state.agent.ainvoke(inputs)

    return AskResponse(response=answer['messages'][-1].content)
