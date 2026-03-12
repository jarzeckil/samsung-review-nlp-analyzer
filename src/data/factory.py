import os

from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


def make_embeddings(
    device='cpu',
    encode_kwargs=None,
) -> Embeddings:
    if encode_kwargs is None:
        encode_kwargs = {'normalize_embeddings': False}

    model_kwargs = {'device': device}

    hf = HuggingFaceEmbeddings(
        model_name=os.getenv('EMBEDDING_MODEL_NAME'),
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    return hf


def make_vector_store(index_name: str, embeddings: Embeddings) -> PineconeVectorStore:
    vector_store = PineconeVectorStore(embedding=embeddings, index_name=index_name)

    return vector_store


def make_model():
    model = ChatGroq(
        model=os.getenv('CHAT_MODEL_NAME'),
        temperature=0.0,
        max_retries=2,
    )

    return model
