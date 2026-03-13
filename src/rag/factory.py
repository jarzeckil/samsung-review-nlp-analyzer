import os
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore

from src.core.config import PROMPT_PATH
from src.data.retrieval import retrieve_from_query


def create_agent_tools(vector_store: PineconeVectorStore) -> list:
    @tool
    def search_semantic_reviews(
        query: str, score: int | None = None, k_results: int = 5
    ) -> str:
        """
        Searches for reviews basing on their content.
        Use when user asks about pros, cons, opinions or reviews.
        After using this tool continue to summary of the results.
        Do not try to conduct more searches.
        Args:
            query: keywords to search
            score: optional score filter. Use only if user explicitly states the
            score value.
            k_results: number of reviews to pull from the database. Use default, unless:
             - the user explicitly states that he wants the answer
             based on bigger/smaller number of reviews.
             - the user asks a more general question that requires a larger
             sample of reviews.
        """
        results = retrieve_from_query(
            vector_store, query, k_results=k_results, score_filter=score
        )

        if not results:
            return 'No reviews found.'

        formatted_results = '\n\n'.join(
            [
                f'Ocena: {doc.metadata["Score"]} | Opinia: {doc.page_content}'
                for doc in results
            ]
        )
        return f'Znalezione opinie:\n{formatted_results}'

    return [search_semantic_reviews]


def make_model():
    local = os.getenv('LOCAL_MODEL')

    if local == 'true':
        model = ChatOllama(
            model=os.getenv('OLLAMA_MODEL_NAME'),
            temperature=0.0,
        )
    else:
        model = ChatGroq(
            model=os.getenv('GROQ_MODEL_NAME'),
            temperature=0.0,
            max_retries=2,
        )

    return model


def make_agent(tools: list):

    model = make_model()

    prompt_template = Path(PROMPT_PATH).read_text(encoding='utf-8')
    return create_agent(model=model, tools=tools, system_prompt=prompt_template)
