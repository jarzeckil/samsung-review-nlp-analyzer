def create_prompt(data: list, query: str) -> str:
    prompt = f"""
    You are an analyst answering a customer's question. Respond basing only on 
    provided client reviews. If the answer is not in the reviews, 
    you should explicitly state that and not respond further.
    Reviews: {data}
    Question: {query}
    """

    return prompt
