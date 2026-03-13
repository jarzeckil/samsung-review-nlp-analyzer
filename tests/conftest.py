"""Shared pytest fixtures and configuration."""

from collections.abc import Generator
from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient
from langchain_core.documents import Document
import pytest


@pytest.fixture
def mock_env_vars(monkeypatch) -> None:
    """Mock environment variables for testing."""
    env_vars = {
        'DEVICE': 'cpu',
        'PINECONE_API_KEY': 'test-pinecone-key',
        'INDEX_NAME': 'test-index',
        'CSV_NAME': 'test.csv',
        'GROQ_API_KEY': 'test-groq-key',
        'GROQ_MODEL_NAME': 'llama-3.1-70b-versatile',
        'HF_TOKEN': 'test-hf-token',
        'EMBEDDING_MODEL_NAME': 'sentence-transformers/all-MiniLM-L6-v2',
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    # Ensure LOCAL_MODEL is not set (use Groq by default)
    monkeypatch.delenv('LOCAL_MODEL', raising=False)


@pytest.fixture
def mock_embeddings() -> Mock:
    """Mock HuggingFace embeddings model."""
    embeddings = Mock()
    embeddings.embed_query.return_value = [0.1] * 768
    embeddings.embed_documents.return_value = [[0.1] * 768]
    return embeddings


@pytest.fixture
def mock_vector_store() -> Mock:
    """Mock Pinecone vector store."""
    vector_store = Mock()
    vector_store.similarity_search.return_value = [
        Document(
            page_content='Test review 1',
            metadata={'Date': '2024-01-01', 'Score': '1'},
        ),
        Document(
            page_content='Test review 2',
            metadata={'Date': '2024-01-02', 'Score': '5'},
        ),
    ]
    vector_store.add_documents.return_value = ['id1', 'id2']
    return vector_store


@pytest.fixture
def mock_llm() -> Mock:
    """Mock ChatGroq LLM."""
    llm = Mock()
    mock_response = Mock()
    mock_response.content = 'This is a test response from the LLM.'
    llm.invoke.return_value = mock_response
    return llm


@pytest.fixture
def sample_documents() -> list[Document]:
    """Sample documents for testing."""
    return [
        Document(
            page_content='Great product, works perfectly!',
            metadata={'Date': '2024-01-01', 'Score': '5'},
        ),
        Document(
            page_content='Terrible delivery, arrived late.',
            metadata={'Date': '2024-01-02', 'Score': '1'},
        ),
        Document(
            page_content='Average experience, nothing special.',
            metadata={'Date': '2024-01-03', 'Score': '3'},
        ),
    ]


@pytest.fixture
def mock_agent() -> AsyncMock:
    """Mock LangChain agent with AsyncMock for proper assertion tracking."""
    agent = AsyncMock()
    mock_response = {
        'messages': [
            Mock(content='user question'),
            Mock(content='agent reasoning'),
            Mock(content='Final answer based on reviews'),
        ]
    }

    # Set ainvoke to return the mock response
    agent.ainvoke.return_value = mock_response
    agent.invoke.return_value = mock_response
    return agent


@pytest.fixture
def app_with_mock_agent(mock_agent, monkeypatch) -> Generator[TestClient]:
    """FastAPI test client with mocked agent."""
    # Mock the factory functions to avoid loading real models
    mock_embeddings = Mock()
    mock_embeddings.embed_query.return_value = [0.1] * 768

    mock_vector_store = Mock()
    mock_vector_store.similarity_search.return_value = []

    mock_tools = []

    # Patch the factory functions before importing app
    monkeypatch.setattr(
        'src.data.factory.make_embeddings', lambda **kwargs: mock_embeddings
    )
    monkeypatch.setattr(
        'src.data.factory.make_vector_store', lambda *args, **kwargs: mock_vector_store
    )
    monkeypatch.setattr(
        'src.rag.factory.create_agent_tools', lambda *args, **kwargs: mock_tools
    )
    monkeypatch.setattr(
        'src.rag.factory.make_agent', lambda *args, **kwargs: mock_agent
    )

    # Set required environment variables
    monkeypatch.setenv('DEVICE', 'cpu')
    monkeypatch.setenv('INDEX_NAME', 'test-index')

    # Import app after patching
    from src.serving.app import app

    with TestClient(app) as client:
        yield client
