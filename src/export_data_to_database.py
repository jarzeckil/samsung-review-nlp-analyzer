import os

from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader

from config import DATA_DIR
from factory import make_embeddings, make_vector_store


def ingest_send_data(csv_path: str):
    embeddings = make_embeddings(model_kwargs={'device': os.getenv('DEVICE')})

    loader = CSVLoader(
        file_path=DATA_DIR / csv_path,
        csv_args={
            'delimiter': ',',
        },
        metadata_columns=['Date', 'Score'],
    )

    docs = loader.load()

    vector_store = make_vector_store(os.getenv('INDEX_NAME'), embeddings)

    vector_store.add_documents(docs)

    return vector_store


if __name__ == '__main__':
    load_dotenv()
    _ = ingest_send_data(DATA_DIR / os.getenv('CSV_NAME'))
