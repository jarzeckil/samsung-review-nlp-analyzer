"""Unit tests for data factory functions."""

from unittest.mock import Mock, patch

import pytest


class TestMakeEmbeddings:
    """Tests for make_embeddings factory function."""

    @patch('src.data.factory.HuggingFaceEmbeddings')
    def test_make_embeddings_default_device(self, mock_hf_embeddings, mock_env_vars):
        """Test embeddings creation with default CPU device."""
        from src.data.factory import make_embeddings

        mock_instance = Mock()
        mock_hf_embeddings.return_value = mock_instance

        result = make_embeddings(device='cpu')

        mock_hf_embeddings.assert_called_once()
        call_kwargs = mock_hf_embeddings.call_args.kwargs
        assert call_kwargs['model_name'] == 'sentence-transformers/all-MiniLM-L6-v2'
        assert call_kwargs['model_kwargs'] == {'device': 'cpu'}
        assert result == mock_instance

    @patch('src.data.factory.HuggingFaceEmbeddings')
    def test_make_embeddings_cuda_device(self, mock_hf_embeddings, mock_env_vars):
        """Test embeddings creation with CUDA device."""
        from src.data.factory import make_embeddings

        mock_instance = Mock()
        mock_hf_embeddings.return_value = mock_instance

        result = make_embeddings(device='cuda')

        call_kwargs = mock_hf_embeddings.call_args.kwargs
        assert call_kwargs['model_kwargs'] == {'device': 'cuda'}

    @patch('src.data.factory.HuggingFaceEmbeddings')
    def test_make_embeddings_custom_encode_kwargs(
        self, mock_hf_embeddings, mock_env_vars
    ):
        """Test embeddings with custom encode kwargs."""
        from src.data.factory import make_embeddings

        mock_instance = Mock()
        mock_hf_embeddings.return_value = mock_instance

        custom_kwargs = {'normalize_embeddings': True, 'batch_size': 32}
        result = make_embeddings(device='cpu', encode_kwargs=custom_kwargs)

        call_kwargs = mock_hf_embeddings.call_args.kwargs
        assert call_kwargs['encode_kwargs'] == custom_kwargs


class TestMakeVectorStore:
    """Tests for make_vector_store factory function."""

    @patch('src.data.factory.PineconeVectorStore')
    def test_make_vector_store_creates_instance(self, mock_pinecone, mock_embeddings):
        """Test vector store creation with valid parameters."""
        from src.data.factory import make_vector_store

        mock_instance = Mock()
        mock_pinecone.return_value = mock_instance

        result = make_vector_store(index_name='test-index', embeddings=mock_embeddings)

        mock_pinecone.assert_called_once_with(
            index_name='test-index', embedding=mock_embeddings
        )
        assert result == mock_instance

    @patch('src.data.factory.PineconeVectorStore')
    def test_make_vector_store_with_different_index(
        self, mock_pinecone, mock_embeddings
    ):
        """Test vector store with different index name."""
        from src.data.factory import make_vector_store

        mock_instance = Mock()
        mock_pinecone.return_value = mock_instance

        result = make_vector_store(
            index_name='production-index', embeddings=mock_embeddings
        )

        call_kwargs = mock_pinecone.call_args.kwargs
        assert call_kwargs['index_name'] == 'production-index'
