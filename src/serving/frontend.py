"""
Streamlit frontend for Samsung Review Analyzer.

This module provides a clean, ChatGPT-like interface for interacting with
the Samsung SmartThings review analysis agent via FastAPI backend.
"""

import os

import requests
import streamlit as st

# ============================================================================
# Configuration
# ============================================================================

# API URL can be configured via environment variable (useful for Docker)
API_URL = os.getenv('API_URL', 'http://localhost:8000/ask')
PAGE_TITLE = 'Samsung Review Agent'
PAGE_ICON = '🤖'

# ============================================================================
# Page Setup
# ============================================================================

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout='wide',
    initial_sidebar_state='expanded',
)

# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    if st.button('🗑️ Clear Chat History', use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ============================================================================
# Main Content
# ============================================================================

# Header
st.title('🤖 Samsung Review Agent')
st.markdown(
    """
    Welcome to the Samsung Review Analyzer! Ask questions about Samsung
    reviews and get AI-powered insights based on real customer feedback.
    """
)

# ============================================================================
# Chat Interface
# ============================================================================

# Initialize chat history in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Chat input
if prompt := st.chat_input('Ask me anything about Samsung reviews...'):
    # Add user message to chat history
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # Display user message
    with st.chat_message('user'):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message('assistant'):
        with st.spinner('🤔 Analyzing reviews...'):
            try:
                # Call the FastAPI backend
                response = requests.post(
                    API_URL,
                    json={'question': prompt},
                    timeout=30,
                )

                # Check if request was successful
                if response.status_code == 200:
                    assistant_response = response.json()['response']
                    st.markdown(assistant_response)

                    # Add assistant response to chat history
                    st.session_state.messages.append(
                        {'role': 'assistant', 'content': assistant_response}
                    )
                else:
                    error_message = (
                        f'⚠️ Error: Received status code {response.status_code} '
                        f'from the backend. Please try again.'
                    )
                    st.error(error_message)

                    # Add error to chat history
                    st.session_state.messages.append(
                        {'role': 'assistant', 'content': error_message}
                    )

            except requests.exceptions.ConnectionError:
                error_message = (
                    '❌ **Connection Error**: Unable to connect to the backend API. '
                    'Please ensure the FastAPI server is running on '
                    f'`{API_URL.rsplit("/", 1)[0]}`.'
                )
                st.error(error_message)

                # Add error to chat history
                st.session_state.messages.append(
                    {'role': 'assistant', 'content': error_message}
                )

            except requests.exceptions.Timeout:
                error_message = (
                    '⏱️ **Timeout Error**: The request took too long to complete. '
                    'Please try again with a simpler question.'
                )
                st.error(error_message)

                # Add error to chat history
                st.session_state.messages.append(
                    {'role': 'assistant', 'content': error_message}
                )

            except Exception as e:
                error_message = (
                    f'❌ **Unexpected Error**: {str(e)}\n\n'
                    'Please try again or contact support if the issue persists.'
                )
                st.error(error_message)

                # Add error to chat history
                st.session_state.messages.append(
                    {'role': 'assistant', 'content': error_message}
                )

# ============================================================================
# Footer
# ============================================================================

st.divider()

st.caption(
    '💡 **Tip**: The agent can analyze sentiment, identify trends, '
    'and provide insights based on Samsung SmartThings customer reviews.'
)
