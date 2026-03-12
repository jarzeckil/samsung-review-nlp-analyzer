"""Unit tests for RAG factory functions and tools."""

from unittest.mock import Mock, patch

import pytest


class TestCreateAgentTools:
    """Tests for create_agent_tools function."""

    def test_creates_tool_list(self, mock_vector_store):
        """Test that create_agent_tools returns a list of tools."""
        from src.rag.factory import create_agent_tools

        tools = create_agent_tools(mock_vector_store)

        assert isinstance(tools, list)
        assert len(tools) == 1
        assert tools[0].name == 'search_semantic_reviews'

    def test_tool_has_correct_signature(self, mock_vector_store):
        """Test that the tool has correct parameters."""
        from src.rag.factory import create_agent_tools

        tools = create_agent_tools(mock_vector_store)
        tool = tools[0]

        # Check tool attributes
        assert hasattr(tool, 'name')
        assert hasattr(tool, 'description')
        assert hasattr(tool, 'func')

    def test_tool_execution_without_score_filter(self, mock_vector_store):
        """Test tool execution without score filtering."""
        from src.rag.factory import create_agent_tools

        mock_vector_store.similarity_search.return_value = [
            Mock(
                page_content='Review 1',
                metadata={'Date': '2024-01-01', 'Score': '5'},
            ),
            Mock(
                page_content='Review 2',
                metadata={'Date': '2024-01-02', 'Score': '3'},
            ),
        ]

        tools = create_agent_tools(mock_vector_store)
        tool = tools[0]

        result = tool.func(query='test query', k_results=2)

        mock_vector_store.similarity_search.assert_called_once_with(
            query='test query', k=2, filter={}
        )
        assert 'Znalezione opinie:' in result
        assert 'Review 1' in result
        assert 'Review 2' in result

    def test_tool_execution_with_score_filter(self, mock_vector_store):
        """Test tool execution with score filtering."""
        from src.rag.factory import create_agent_tools

        mock_vector_store.similarity_search.return_value = [
            Mock(page_content='Review 1', metadata={'Date': '2024-01-01', 'Score': '1'})
        ]

        tools = create_agent_tools(mock_vector_store)
        tool = tools[0]

        result = tool.func(query='bad reviews', score=1, k_results=5)

        mock_vector_store.similarity_search.assert_called_once_with(
            query='bad reviews', k=5, filter={'Score': '1'}
        )
        assert 'Ocena: 1' in result

    def test_tool_formats_results_correctly(self, mock_vector_store):
        """Test that tool formats results in expected format."""
        from src.rag.factory import create_agent_tools

        mock_vector_store.similarity_search.return_value = [
            Mock(
                page_content='Great product!',
                metadata={'Date': '2024-01-01', 'Score': '5'},
            )
        ]

        tools = create_agent_tools(mock_vector_store)
        tool = tools[0]

        result = tool.func(query='test', k_results=1)

        assert result.startswith('Znalezione opinie:')
        assert 'Ocena: 5' in result
        assert 'Opinia: Great product!' in result


class TestMakeModel:
    """Tests for make_model function."""

    @patch('src.rag.factory.ChatGroq')
    def test_make_model_creates_groq_instance(self, mock_groq, mock_env_vars):
        """Test that make_model creates ChatGroq with correct config."""
        from src.rag.factory import make_model

        mock_instance = Mock()
        mock_groq.return_value = mock_instance

        result = make_model()

        mock_groq.assert_called_once_with(
            model='llama-3.1-70b-versatile', temperature=0.0, max_retries=2
        )
        assert result == mock_instance

    @patch('src.rag.factory.ChatGroq')
    def test_make_model_uses_env_model_name(self, mock_groq, monkeypatch):
        """Test that make_model uses model name from environment."""
        monkeypatch.setenv('CHAT_MODEL_NAME', 'custom-model-name')

        from src.rag.factory import make_model

        make_model()

        call_kwargs = mock_groq.call_args.kwargs
        assert call_kwargs['model'] == 'custom-model-name'

    @patch('src.rag.factory.ChatGroq')
    def test_make_model_temperature_is_zero(self, mock_groq, mock_env_vars):
        """Test that temperature is set to 0 for deterministic responses."""
        from src.rag.factory import make_model

        make_model()

        call_kwargs = mock_groq.call_args.kwargs
        assert call_kwargs['temperature'] == 0.0


class TestMakeAgent:
    """Tests for make_agent function."""

    @patch('src.rag.factory.make_model')
    @patch('src.rag.factory.create_agent')
    @patch('src.rag.factory.Path')
    def test_make_agent_creates_executor(
        self, mock_path, mock_create_agent, mock_make_model
    ):
        """Test that make_agent creates agent with create_agent."""
        from src.rag.factory import make_agent

        # Mock Path.read_text for system prompt
        mock_path_instance = Mock()
        mock_path_instance.read_text.return_value = 'System prompt text'
        mock_path.return_value = mock_path_instance

        mock_llm = Mock()
        mock_make_model.return_value = mock_llm

        mock_agent_instance = Mock()
        mock_create_agent.return_value = mock_agent_instance

        mock_tools = [Mock()]
        result = make_agent(mock_tools)

        mock_make_model.assert_called_once()
        mock_create_agent.assert_called_once()
        assert result == mock_agent_instance

    @patch('src.rag.factory.make_model')
    @patch('src.rag.factory.create_agent')
    @patch('src.rag.factory.Path')
    def test_make_agent_loads_system_prompt(
        self, mock_path, mock_create_agent, mock_make_model
    ):
        """Test that make_agent loads system prompt from file."""
        from src.rag.factory import make_agent

        # Mock Path.read_text for system prompt
        mock_path_instance = Mock()
        mock_path_instance.read_text.return_value = (
            'You are an analyst answering questions.'
        )
        mock_path.return_value = mock_path_instance

        mock_llm = Mock()
        mock_make_model.return_value = mock_llm

        mock_create_agent.return_value = Mock()

        mock_tools = [Mock()]
        make_agent(mock_tools)

        # Verify Path was called and read_text was called
        mock_path.assert_called_once()
        mock_path_instance.read_text.assert_called_once()

    @patch('src.rag.factory.make_model')
    @patch('src.rag.factory.create_agent')
    @patch('src.rag.factory.Path')
    def test_make_agent_passes_tools_to_agent(
        self, mock_path, mock_create_agent, mock_make_model
    ):
        """Test that make_agent passes tools to the agent."""
        from src.rag.factory import make_agent

        # Mock Path.read_text for system prompt
        mock_path_instance = Mock()
        mock_path_instance.read_text.return_value = 'Prompt'
        mock_path.return_value = mock_path_instance

        mock_llm = Mock()
        mock_make_model.return_value = mock_llm

        mock_create_agent.return_value = Mock()

        mock_tools = [Mock(name='tool1'), Mock(name='tool2')]
        make_agent(mock_tools)

        # Verify tools are passed to agent creation
        call_kwargs = mock_create_agent.call_args.kwargs
        assert 'tools' in call_kwargs
        assert call_kwargs['tools'] == mock_tools
