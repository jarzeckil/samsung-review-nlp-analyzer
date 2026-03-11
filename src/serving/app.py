from contextlib import asynccontextmanager
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, status
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.core.config import PROMPT_PATH
from src.data.factory import make_embeddings, make_model, make_vector_store
from src.serving.schemas import AskRequest, AskResponse

artifacts = {}


@asynccontextmanager
async def lifespan(a: FastAPI):
    load_dotenv()
    embedding = make_embeddings(device=os.getenv('DEVICE'))
    retriever = make_vector_store(os.getenv('INDEX_NAME'), embedding).as_retriever(k=5)
    model: BaseChatModel = make_model()
    prompt_template = Path(PROMPT_PATH).read_text(encoding='utf-8')
    prompt = ChatPromptTemplate.from_template(prompt_template)

    rag_chain = (
        {'context': retriever, 'query': RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    artifacts['chain'] = rag_chain

    yield


app = FastAPI(lifespan=lifespan)


@app.post('/ask', status_code=status.HTTP_200_OK)
async def ask(query: AskRequest):
    answer = await artifacts['chain'].ainvoke(query.question)

    return AskResponse(response=answer)
