# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from unittest.mock import patch

from litellm.types.utils import Choices, EmbeddingResponse, Message, ModelResponse, TextCompletionResponse, Usage
import pytest

from data_designer.config.models import ModelType
from data_designer.engine.models.errors import ModelGenerationValidationFailureError
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.models.parsers.errors import ParserException

MockMessage = namedtuple("MockMessage", ["content"])
MockChoice = namedtuple("MockChoice", ["message"])
MockCompletion = namedtuple("MockCompletion", ["choices"])


def mock_oai_response_object(response_text: str) -> MockCompletion:
    return MockCompletion(choices=[MockChoice(message=MockMessage(content=response_text))])


@pytest.fixture
def stub_model_facade(stub_model_configs, stub_secrets_resolver, stub_model_provider_registry):
    return ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
    )


@pytest.fixture
def stub_expected_response():
    return ModelResponse(choices=Choices(message=Message(content="Test response")))


@pytest.mark.parametrize(
    "max_correction_steps,max_conversation_restarts,total_calls",
    [
        (0, 0, 1),
        (1, 1, 4),
        (1, 2, 6),
        (5, 0, 6),
        (0, 5, 6),
        (3, 3, 16),
    ],
)
@patch("data_designer.engine.models.facade.ModelFacade.completion", autospec=True)
def test_generate(
    mock_completion,
    stub_model_facade,
    max_correction_steps,
    max_conversation_restarts,
    total_calls,
):
    bad_response = mock_oai_response_object("bad response")
    mock_completion.side_effect = lambda *args, **kwargs: bad_response

    def _failing_parser(response: str):
        raise ParserException("parser exception")

    with pytest.raises(ModelGenerationValidationFailureError):
        stub_model_facade.generate(
            prompt="foo",
            system_prompt="bar",
            parser=_failing_parser,
            max_correction_steps=max_correction_steps,
            max_conversation_restarts=max_conversation_restarts,
        )
    assert mock_completion.call_count == total_calls

    with pytest.raises(ModelGenerationValidationFailureError):
        stub_model_facade.generate(
            prompt="foo",
            parser=_failing_parser,
            system_prompt="bar",
            max_correction_steps=max_correction_steps,
            max_conversation_restarts=max_conversation_restarts,
        )
    assert mock_completion.call_count == 2 * total_calls


@pytest.mark.parametrize(
    "system_prompt,expected_messages",
    [
        ("", [{"role": "user", "content": "does not matter"}]),
        ("hello!", [{"content": "hello!", "role": "system"}, {"role": "user", "content": "does not matter"}]),
    ],
)
@patch("data_designer.engine.models.facade.ModelFacade.completion", autospec=True)
def test_generate_with_system_prompt(mock_completion, stub_model_facade, system_prompt, expected_messages):
    mock_completion.return_value = ModelResponse(choices=Choices(message=Message(content="Hello!")))

    stub_model_facade.generate(prompt="does not matter", system_prompt=system_prompt, parser=lambda x: x)
    assert mock_completion.call_count == 1
    assert mock_completion.call_args[0][1] == expected_messages


def test_model_alias_property(stub_model_facade, stub_model_configs):
    assert stub_model_facade.model_alias == stub_model_configs[0].alias


def test_usage_stats_property(stub_model_facade):
    assert stub_model_facade.usage_stats is not None
    assert hasattr(stub_model_facade.usage_stats, "model_dump")


@pytest.mark.parametrize(
    "skip_usage_tracking",
    [
        False,
        True,
    ],
)
def test_completion_success(stub_model_facade, stub_expected_response, skip_usage_tracking):
    stub_model_facade._router.completion = lambda model_name, messages, **kwargs: stub_expected_response

    messages = [{"role": "user", "content": "test"}]
    result = stub_model_facade.completion(messages, skip_usage_tracking=skip_usage_tracking)

    assert result == stub_expected_response


def test_completion_with_exception(stub_model_facade):
    def raise_exception(*args, **kwargs):
        raise Exception("Router error")

    stub_model_facade._router.completion = raise_exception

    messages = [{"role": "user", "content": "test"}]

    with pytest.raises(Exception, match="Router error"):
        stub_model_facade.completion(messages)


def test_completion_with_kwargs(stub_model_facade, stub_expected_response):
    captured_kwargs = {}

    def mock_completion(model_name, messages, **kwargs):
        captured_kwargs.update(kwargs)
        return stub_expected_response

    stub_model_facade._router.completion = mock_completion

    messages = [{"role": "user", "content": "test"}]
    kwargs = {"temperature": 0.7, "max_tokens": 100}
    result = stub_model_facade.completion(messages, **kwargs)

    assert result == stub_expected_response
    assert captured_kwargs == kwargs


@patch("data_designer.engine.models.facade.CustomRouter.completion", autospec=True)
def test_completion_with_extra_body(mock_router_completion, stub_model_facade):
    messages = [{"role": "user", "content": "test"}]

    # completion call has no extra body argument and provider has no extra body
    _ = stub_model_facade.completion(messages)
    assert len(mock_router_completion.call_args) == 2
    assert mock_router_completion.call_args[0][1] == "stub-model-text"
    assert mock_router_completion.call_args[0][2] == messages

    # completion call has no extra body argument and provider has extra body.
    # Should pull extra body from model provider
    custom_extra_body = {"some_custom_key": "some_custom_value"}
    stub_model_facade.model_provider.extra_body = custom_extra_body
    _ = stub_model_facade.completion(messages)
    assert mock_router_completion.call_args[1] == {"extra_body": custom_extra_body}

    # completion call has extra body argument and provider has extra body.
    # Should merge the two with provider extra body taking precedence
    completion_extra_body = {"some_completion_key": "some_completion_value", "some_custom_key": "some_different_value"}
    _ = stub_model_facade.completion(messages, extra_body=completion_extra_body)
    assert mock_router_completion.call_args[1] == {"extra_body": {**completion_extra_body, **custom_extra_body}}


def test_model_type_property(stub_model_facade):
    assert stub_model_facade.model_type == ModelType.CHAT


@pytest.mark.parametrize(
    "skip_usage_tracking",
    [
        False,
        True,
    ],
)
def test_text_completion_success(stub_model_facade, skip_usage_tracking):
    expected_text = "This is a test completion"
    stub_response = TextCompletionResponse(
        choices=[{"text": expected_text, "index": 0, "finish_reason": "stop"}],
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )
    stub_model_facade._router.text_completion = lambda model_name, prompt, **kwargs: stub_response

    prompt = "Complete this sentence:"
    result = stub_model_facade.text_completion(prompt, skip_usage_tracking=skip_usage_tracking)

    assert result == stub_response
    assert result.choices[0]["text"] == expected_text


def test_text_completion_with_exception(stub_model_facade):
    def raise_exception(*args, **kwargs):
        raise Exception("Router error")

    stub_model_facade._router.text_completion = raise_exception

    with pytest.raises(Exception, match="Router error"):
        stub_model_facade.text_completion("test prompt")


def test_text_completion_with_kwargs(stub_model_facade):
    captured_kwargs = {}
    expected_text = "Completed text"
    stub_response = TextCompletionResponse(
        choices=[{"text": expected_text, "index": 0, "finish_reason": "stop"}],
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )

    def mock_text_completion(model_name, prompt, **kwargs):
        captured_kwargs.update(kwargs)
        return stub_response

    stub_model_facade._router.text_completion = mock_text_completion

    kwargs = {"temperature": 0.8, "max_tokens": 150}
    result = stub_model_facade.text_completion("test prompt", **kwargs)

    assert result == stub_response
    assert captured_kwargs == kwargs


@pytest.mark.parametrize(
    "skip_usage_tracking",
    [
        False,
        True,
    ],
)
def test_embedding_success_single_input(stub_model_facade, skip_usage_tracking):
    expected_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    stub_response = EmbeddingResponse(
        data=[{"embedding": expected_embedding, "index": 0, "object": "embedding"}],
        model="test-model",
        usage=Usage(prompt_tokens=5, total_tokens=5),
    )
    stub_model_facade._router.embedding = lambda model, input, **kwargs: stub_response

    input_text = "Test embedding input"
    result = stub_model_facade.embedding(input_text, skip_usage_tracking=skip_usage_tracking)

    assert result == stub_response
    assert result.data[0]["embedding"] == expected_embedding


@pytest.mark.parametrize(
    "skip_usage_tracking",
    [
        False,
        True,
    ],
)
def test_embedding_success_multiple_inputs(stub_model_facade, skip_usage_tracking):
    expected_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    stub_response = EmbeddingResponse(
        data=[
            {"embedding": expected_embeddings[0], "index": 0, "object": "embedding"},
            {"embedding": expected_embeddings[1], "index": 1, "object": "embedding"},
        ],
        model="test-model",
        usage=Usage(prompt_tokens=10, total_tokens=10),
    )
    stub_model_facade._router.embedding = lambda model, input, **kwargs: stub_response

    input_texts = ["First text", "Second text"]
    result = stub_model_facade.embedding(input_texts, skip_usage_tracking=skip_usage_tracking)

    assert result == stub_response
    assert len(result.data) == 2
    assert result.data[0]["embedding"] == expected_embeddings[0]
    assert result.data[1]["embedding"] == expected_embeddings[1]


def test_embedding_with_exception(stub_model_facade):
    def raise_exception(*args, **kwargs):
        raise Exception("Embedding error")

    stub_model_facade._router.embedding = raise_exception

    with pytest.raises(Exception, match="Embedding error"):
        stub_model_facade.embedding("test input")


def test_embedding_with_kwargs(stub_model_facade):
    captured_kwargs = {}
    stub_response = EmbeddingResponse(
        data=[{"embedding": [0.1, 0.2], "index": 0, "object": "embedding"}],
        model="test-model",
        usage=Usage(prompt_tokens=5, total_tokens=5),
    )

    def mock_embedding(model, input, **kwargs):
        captured_kwargs.update(kwargs)
        return stub_response

    stub_model_facade._router.embedding = mock_embedding

    kwargs = {"dimensions": 512}
    result = stub_model_facade.embedding("test input", **kwargs)

    assert result == stub_response
    assert "dimensions" in captured_kwargs
    assert captured_kwargs["dimensions"] == 512


def test_embedding_usage_tracking(stub_model_facade):
    stub_response = EmbeddingResponse(
        data=[{"embedding": [0.1, 0.2], "index": 0, "object": "embedding"}],
        model="test-model",
        usage=Usage(prompt_tokens=10, total_tokens=10),
    )
    stub_model_facade._router.embedding = lambda model, input, **kwargs: stub_response

    initial_prompt_tokens = stub_model_facade.usage_stats.token_usage.prompt_tokens
    initial_completion_tokens = stub_model_facade.usage_stats.token_usage.completion_tokens

    stub_model_facade.embedding("test input", skip_usage_tracking=False)

    assert stub_model_facade.usage_stats.token_usage.prompt_tokens == initial_prompt_tokens + 10
    assert stub_model_facade.usage_stats.token_usage.completion_tokens == initial_completion_tokens + 0


def test_text_completion_usage_tracking(stub_model_facade):
    stub_response = TextCompletionResponse(
        choices=[{"text": "test", "index": 0, "finish_reason": "stop"}],
        usage=Usage(prompt_tokens=5, completion_tokens=15, total_tokens=20),
    )
    stub_model_facade._router.text_completion = lambda model_name, prompt, **kwargs: stub_response

    initial_prompt_tokens = stub_model_facade.usage_stats.token_usage.prompt_tokens
    initial_completion_tokens = stub_model_facade.usage_stats.token_usage.completion_tokens

    stub_model_facade.text_completion("test prompt", skip_usage_tracking=False)

    assert stub_model_facade.usage_stats.token_usage.prompt_tokens == initial_prompt_tokens + 5
    assert stub_model_facade.usage_stats.token_usage.completion_tokens == initial_completion_tokens + 15
