"""Unit tests for data retrieval functions."""

from langchain_core.documents import Document


class TestRetrieveFromQuery:
    """Tests for retrieve_from_query function."""

    def test_retrieve_without_filter(self, mock_vector_store):
        """Test basic retrieval without score filter."""
        from src.data.retrieval import retrieve_from_query

        mock_vector_store.similarity_search.return_value = [
            Document(
                page_content='Review text 1',
                metadata={'Date': '2024-01-01', 'Score': '5'},
            ),
            Document(
                page_content='Review text 2',
                metadata={'Date': '2024-01-02', 'Score': '3'},
            ),
        ]

        results = retrieve_from_query(
            vector_store=mock_vector_store,
            query='test query',
            k_results=2,
            score_filter=None,
        )

        mock_vector_store.similarity_search.assert_called_once_with(
            query='test query', k=2, filter={}
        )
        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)

    def test_retrieve_with_score_filter(self, mock_vector_store):
        """Test retrieval with score filtering."""
        from src.data.retrieval import retrieve_from_query

        mock_vector_store.similarity_search.return_value = [
            Document(
                page_content='Bad review',
                metadata={'Date': '2024-01-01', 'Score': '1'},
            )
        ]

        results = retrieve_from_query(
            vector_store=mock_vector_store,
            query='complaints',
            k_results=5,
            score_filter=1,
        )

        mock_vector_store.similarity_search.assert_called_once_with(
            query='complaints', k=5, filter={'Score': '1'}
        )
        assert len(results) == 1
        assert results[0].metadata['Score'] == '1'

    def test_retrieve_with_different_k_values(self, mock_vector_store):
        """Test retrieval with different k_results values."""
        from src.data.retrieval import retrieve_from_query

        # Create 10 mock documents
        mock_docs = [
            Document(
                page_content=f'Review {i}',
                metadata={'Date': '2024-01-01', 'Score': '5'},
            )
            for i in range(10)
        ]
        mock_vector_store.similarity_search.return_value = mock_docs[:3]

        results = retrieve_from_query(
            vector_store=mock_vector_store,
            query='test',
            k_results=3,
            score_filter=None,
        )

        call_kwargs = mock_vector_store.similarity_search.call_args.kwargs
        assert call_kwargs['k'] == 3
        assert len(results) == 3

    def test_retrieve_returns_empty_list_when_no_results(self, mock_vector_store):
        """Test that empty results are handled correctly."""
        from src.data.retrieval import retrieve_from_query

        mock_vector_store.similarity_search.return_value = []

        results = retrieve_from_query(
            vector_store=mock_vector_store,
            query='nonexistent',
            k_results=5,
            score_filter=None,
        )

        assert results == []
        assert isinstance(results, list)

    def test_retrieve_preserves_metadata(self, mock_vector_store):
        """Test that document metadata is preserved."""
        from src.data.retrieval import retrieve_from_query

        expected_metadata = {
            'Date': '2024-03-15',
            'Score': '4',
            'custom_field': 'value',
        }
        mock_vector_store.similarity_search.return_value = [
            Document(page_content='Test', metadata=expected_metadata)
        ]

        results = retrieve_from_query(
            vector_store=mock_vector_store,
            query='test',
            k_results=1,
            score_filter=None,
        )

        assert results[0].metadata == expected_metadata
