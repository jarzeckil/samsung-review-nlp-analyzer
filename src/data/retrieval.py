from langchain_pinecone import PineconeVectorStore


def retrieve_from_query(
    vector_store: PineconeVectorStore,
    query: str,
    k_results: int,
    score_filter: int | None,
) -> list:
    if score_filter is not None:
        flt = {'Score': f'{score_filter}'}
    else:
        flt = {}

    results = vector_store.similarity_search(query=query, k=k_results, filter=flt)

    return results
