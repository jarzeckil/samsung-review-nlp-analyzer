from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


def make_embeddings(
    model_name: str = 'sentence-transformers/all-mpnet-base-v2',
    model_kwargs=None,
    encode_kwargs=None,
) -> Embeddings:
    if encode_kwargs is None:
        encode_kwargs = {'normalize_embeddings': False}
    if model_kwargs is None:
        model_kwargs = {'device': 'cpu'}

    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    return hf


def make_vector_store(index_name: str, embeddings: Embeddings) -> PineconeVectorStore:
    vector_store = PineconeVectorStore(embedding=embeddings, index_name=index_name)

    return vector_store
