"""Tests for the format converter."""

from local_llm_sim.core.converter import (
    ollama_messages_to_openai,
    openai_chunk_to_ollama,
    openai_response_to_ollama,
)
from local_llm_sim.models.ollama import OllamaMessage


class TestOllamaToOpenAIConversion:
    """Tests for Ollama -> OpenAI message conversion."""

    def test_simple_text_message(self) -> None:
        """Test converting a simple text message."""
        messages = [
            OllamaMessage(role="user", content="Hello, world!"),
        ]

        result = ollama_messages_to_openai(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello, world!"

    def test_system_and_user_messages(self) -> None:
        """Test converting system and user messages."""
        messages = [
            OllamaMessage(role="system", content="You are a helpful assistant."),
            OllamaMessage(role="user", content="What is 2+2?"),
        ]

        result = ollama_messages_to_openai(messages)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a helpful assistant."
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "What is 2+2?"

    def test_assistant_message(self) -> None:
        """Test converting an assistant message."""
        messages = [
            OllamaMessage(role="assistant", content="Hello! How can I help?"),
        ]

        result = ollama_messages_to_openai(messages)

        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Hello! How can I help?"

    def test_image_message(self) -> None:
        """Test converting a message with images."""
        messages = [
            OllamaMessage(
                role="user",
                content="What's in this image?",
                images=["base64encodedimage"],
            ),
        ]

        result = ollama_messages_to_openai(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0] == {"type": "text", "text": "What's in this image?"}
        assert result[0]["content"][1]["type"] == "image_url"
        assert (
            "data:image/jpeg;base64,base64encodedimage"
            in result[0]["content"][1]["image_url"]["url"]
        )

    def test_image_with_data_uri(self) -> None:
        """Test that images with data URI are preserved."""
        messages = [
            OllamaMessage(
                role="user",
                content="Analyze this",
                images=["data:image/png;base64,pngdata"],
            ),
        ]

        result = ollama_messages_to_openai(messages)

        # Should preserve the data URI
        assert result[0]["content"][1]["image_url"]["url"] == "data:image/png;base64,pngdata"

    def test_tool_calls_message(self) -> None:
        """Test converting a message with tool calls."""
        messages = [
            OllamaMessage(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "London"}',
                        },
                    }
                ],
            ),
        ]

        result = ollama_messages_to_openai(messages)

        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == ""
        assert "tool_calls" in result[0]
        assert len(result[0]["tool_calls"]) == 1

    def test_empty_content(self) -> None:
        """Test handling empty content."""
        messages = [
            OllamaMessage(role="assistant", content=""),
        ]

        result = ollama_messages_to_openai(messages)

        assert result[0]["content"] == ""


class TestOpenAIChunkToOllama:
    """Tests for OpenAI streaming chunk -> Ollama conversion."""

    def test_content_chunk(self) -> None:
        """Test converting a content chunk."""
        chunk = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "Hello",
                    },
                    "finish_reason": None,
                }
            ],
        }

        result = openai_chunk_to_ollama(chunk, "test-model")

        assert result["model"] == "test-model"
        assert result["message"]["content"] == "Hello"
        assert result["done"] is False

    def test_role_chunk(self) -> None:
        """Test converting a role chunk."""
        chunk = {
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                    },
                    "finish_reason": None,
                }
            ],
        }

        result = openai_chunk_to_ollama(chunk, "test-model")

        assert result["message"]["role"] == "assistant"
        assert result["done"] is False

    def test_finish_chunk(self) -> None:
        """Test converting a finish chunk."""
        chunk = {
            "choices": [
                {
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }

        result = openai_chunk_to_ollama(chunk, "test-model")

        assert result["done"] is True
        assert result["done_reason"] == "stop"

    def test_tool_call_chunk(self) -> None:
        """Test converting a chunk with tool calls."""
        chunk = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_123",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        }

        result = openai_chunk_to_ollama(chunk, "test-model")

        assert "tool_calls" in result["message"]

    def test_empty_choices(self) -> None:
        """Test handling empty choices array."""
        chunk = {"choices": []}

        result = openai_chunk_to_ollama(chunk, "test-model")

        assert result["message"]["content"] == ""
        assert result["done"] is False


class TestOpenAIResponseToOllama:
    """Tests for non-streaming OpenAI response -> Ollama conversion."""

    def test_basic_response(self) -> None:
        """Test converting a basic response."""
        response = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        result = openai_response_to_ollama(response, "test-model")

        assert result["model"] == "test-model"
        assert result["message"]["role"] == "assistant"
        assert result["message"]["content"] == "Hello! How can I help?"
        assert result["done"] is True
        assert result["done_reason"] == "stop"
        assert result["prompt_eval_count"] == 10
        assert result["eval_count"] == 5

    def test_response_with_tool_calls(self) -> None:
        """Test converting a response with tool calls."""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "London"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }

        result = openai_response_to_ollama(response, "test-model")

        assert result["message"]["content"] == ""
        assert "tool_calls" in result["message"]
        assert result["done_reason"] == "tool_calls"

    def test_empty_usage(self) -> None:
        """Test handling missing usage data."""
        response = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop",
                }
            ],
        }

        result = openai_response_to_ollama(response, "test-model")

        assert result["prompt_eval_count"] == 0
        assert result["eval_count"] == 0
