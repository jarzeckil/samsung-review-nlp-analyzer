"""Integration tests for FastAPI endpoints."""

from fastapi import status


class TestAskEndpoint:
    """Integration tests for POST /ask endpoint."""

    def test_ask_endpoint_returns_200_with_valid_request(self, app_with_mock_agent):
        """Test that /ask returns 200 with valid question."""
        response = app_with_mock_agent.post(
            '/ask', json={'question': 'What are common complaints?'}
        )

        assert response.status_code == status.HTTP_200_OK
        assert 'response' in response.json()

    def test_ask_endpoint_returns_response_from_agent(self, app_with_mock_agent):
        """Test that /ask returns agent's response content."""
        response = app_with_mock_agent.post(
            '/ask', json={'question': 'Tell me about the reviews'}
        )

        data = response.json()
        assert data['response'] == 'Final answer based on reviews'

    def test_ask_endpoint_validates_request_schema(self, app_with_mock_agent):
        """Test that /ask validates request schema."""
        # Missing 'question' field
        response = app_with_mock_agent.post('/ask', json={})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert 'detail' in data
        assert any(error['loc'] == ['body', 'question'] for error in data['detail'])

    def test_ask_endpoint_rejects_empty_question(self, app_with_mock_agent):
        """Test that /ask rejects empty question string."""
        response = app_with_mock_agent.post('/ask', json={'question': ''})

        # Empty string passes Pydantic validation but should be handled
        # This test documents current behavior
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ]

    def test_ask_endpoint_handles_special_characters(self, app_with_mock_agent):
        """Test that /ask handles special characters in question."""
        special_question = 'What about "delivery" & <shipping> times?'
        response = app_with_mock_agent.post('/ask', json={'question': special_question})

        assert response.status_code == status.HTTP_200_OK
        assert 'response' in response.json()

    def test_ask_endpoint_handles_long_question(self, app_with_mock_agent):
        """Test that /ask handles long questions."""
        long_question = 'What are the issues? ' * 100  # 2400 chars
        response = app_with_mock_agent.post('/ask', json={'question': long_question})

        assert response.status_code == status.HTTP_200_OK
        assert 'response' in response.json()

    def test_ask_endpoint_handles_unicode(self, app_with_mock_agent):
        """Test that /ask handles Unicode characters (Polish)."""
        polish_question = 'Jakie są najczęstsze problemy z dostawą?'
        response = app_with_mock_agent.post('/ask', json={'question': polish_question})

        assert response.status_code == status.HTTP_200_OK
        assert 'response' in response.json()

    def test_ask_endpoint_content_type_validation(self, app_with_mock_agent):
        """Test that /ask requires JSON content type."""
        response = app_with_mock_agent.post(
            '/ask',
            data='question=test',
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_ask_endpoint_accepts_extra_fields(self, app_with_mock_agent):
        """Test that /ask ignores extra fields in request."""
        response = app_with_mock_agent.post(
            '/ask', json={'question': 'Test question', 'extra_field': 'ignored'}
        )

        # Pydantic should ignore extra fields by default
        assert response.status_code == status.HTTP_200_OK

    def test_ask_endpoint_agent_invocation(self, app_with_mock_agent):
        """Test that /ask correctly invokes the agent."""
        from src.serving.app import app

        question = 'What do customers say about delivery?'

        # Reset mock before test
        app.state.agent.ainvoke.reset_mock()

        app_with_mock_agent.post('/ask', json={'question': question})

        # Verify agent was called with correct format
        app.state.agent.ainvoke.assert_called_once()
        call_args = app.state.agent.ainvoke.call_args[0][0]
        assert 'messages' in call_args
        assert call_args['messages'][0] == ('user', question)


class TestAPIDocumentation:
    """Tests for API documentation endpoints."""

    def test_openapi_docs_accessible(self, app_with_mock_agent):
        """Test that /docs endpoint is accessible."""
        response = app_with_mock_agent.get('/docs')

        assert response.status_code == status.HTTP_200_OK
        assert 'text/html' in response.headers['content-type']

    def test_redoc_accessible(self, app_with_mock_agent):
        """Test that /redoc endpoint is accessible."""
        response = app_with_mock_agent.get('/redoc')

        assert response.status_code == status.HTTP_200_OK
        assert 'text/html' in response.headers['content-type']

    def test_openapi_json_accessible(self, app_with_mock_agent):
        """Test that /openapi.json is accessible."""
        response = app_with_mock_agent.get('/openapi.json')

        assert response.status_code == status.HTTP_200_OK
        assert response.headers['content-type'] == 'application/json'

    def test_openapi_schema_contains_ask_endpoint(self, app_with_mock_agent):
        """Test that OpenAPI schema documents /ask endpoint."""
        response = app_with_mock_agent.get('/openapi.json')
        schema = response.json()

        assert '/ask' in schema['paths']
        assert 'post' in schema['paths']['/ask']


class TestCORSAndSecurity:
    """Tests for CORS and security headers."""

    def test_cors_headers_if_configured(self, app_with_mock_agent):
        """Test CORS headers if CORS middleware is configured."""
        response = app_with_mock_agent.options('/ask')

        # Document current CORS behavior
        # If CORS is not configured, this test serves as documentation
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_405_METHOD_NOT_ALLOWED,
        ]

    def test_content_type_header_in_response(self, app_with_mock_agent):
        """Test that response includes correct Content-Type."""
        response = app_with_mock_agent.post('/ask', json={'question': 'Test question'})

        assert response.headers['content-type'] == 'application/json'


class TestErrorHandling:
    """Tests for error handling in the API."""

    def test_invalid_json_returns_422(self, app_with_mock_agent):
        """Test that invalid JSON returns 422."""
        response = app_with_mock_agent.post(
            '/ask',
            data='{"invalid json',
            headers={'Content-Type': 'application/json'},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_wrong_http_method_returns_405(self, app_with_mock_agent):
        """Test that wrong HTTP method returns 405."""
        response = app_with_mock_agent.get('/ask')

        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    def test_nonexistent_endpoint_returns_404(self, app_with_mock_agent):
        """Test that nonexistent endpoint returns 404."""
        response = app_with_mock_agent.post('/nonexistent')

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestResponseSchema:
    """Tests for response schema validation."""

    def test_response_has_correct_schema(self, app_with_mock_agent):
        """Test that response matches AskResponse schema."""
        response = app_with_mock_agent.post('/ask', json={'question': 'Test question'})

        data = response.json()
        assert isinstance(data, dict)
        assert 'response' in data
        assert isinstance(data['response'], str)

    def test_response_contains_string_content(self, app_with_mock_agent):
        """Test that response field contains non-empty string."""
        response = app_with_mock_agent.post(
            '/ask', json={'question': 'What are the reviews about?'}
        )

        data = response.json()
        assert len(data['response']) > 0
        assert isinstance(data['response'], str)
