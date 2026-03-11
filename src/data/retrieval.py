from langchain_pinecone import PineconeVectorStore


def retrieve_from_query(
    vector_store: PineconeVectorStore, query: str, k_results: int
) -> list:
    results = vector_store.similarity_search(query=query, k=k_results)

    return results
