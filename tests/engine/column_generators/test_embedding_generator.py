# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from litellm.types.utils import EmbeddingResponse, Usage
import numpy as np
import pytest

from data_designer.config.column_configs import EmbeddingColumnConfig
from data_designer.config.models import InferenceParameters, ModelConfig
from data_designer.engine.column_generators.generators.llm_generators import EmbeddingColumnGenerator
from data_designer.engine.resources.resource_provider import ResourceProvider


@pytest.fixture
def mock_model_facade():
    """Create a mock ModelFacade."""
    facade = MagicMock()
    facade.embedding = MagicMock()
    return facade


@pytest.fixture
def mock_model_registry(mock_model_facade):
    """Create a mock ModelRegistry."""
    registry = MagicMock()
    registry.get_model.return_value = mock_model_facade
    registry.get_model_config.return_value = ModelConfig(
        alias="embed-model",
        model="test-embedding-model",
        inference_parameters=InferenceParameters(),
    )
    registry.get_model_provider.return_value = MagicMock(name="test-provider")
    return registry


@pytest.fixture
def mock_resource_provider(mock_model_registry):
    """Create a mock ResourceProvider."""
    provider = MagicMock(spec=ResourceProvider)
    provider.model_registry = mock_model_registry
    return provider


@pytest.fixture
def embedding_config():
    """Create a basic embedding column config."""
    return EmbeddingColumnConfig(
        name="text_embedding",
        model_alias="embed-model",
        input_text="{{product_name}}",
    )


@pytest.fixture
def embedding_generator(embedding_config, mock_resource_provider):
    """Create an embedding column generator."""
    generator = EmbeddingColumnGenerator(
        config=embedding_config,
        resource_provider=mock_resource_provider,
    )
    return generator


def test_embedding_generator_metadata():
    """Test embedding generator metadata."""
    metadata = EmbeddingColumnGenerator.metadata()

    assert metadata.name == "embedding_generator"
    assert metadata.description == "Generate embedding vectors for text inputs"
    assert metadata.generation_strategy == "cell_by_cell"


def test_embedding_generator_basic(embedding_generator, mock_model_facade):
    """Test basic embedding generation."""
    # Setup mock response
    embedding_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_response = EmbeddingResponse(
        data=[{"embedding": embedding_vector, "index": 0, "object": "embedding"}],
        model="test-model",
        usage=Usage(prompt_tokens=5, total_tokens=5),
    )
    mock_model_facade.embedding.return_value = mock_response

    # Generate embedding
    data = {"product_name": "Widget"}
    result = embedding_generator.generate(data)

    # Verify
    assert "text_embedding" in result
    assert result["text_embedding"] == embedding_vector
    mock_model_facade.embedding.assert_called_once()


def test_embedding_generator_with_normalization(mock_resource_provider, mock_model_facade):
    """Test embedding generation with normalization."""
    config = EmbeddingColumnConfig(
        name="normalized_embedding",
        model_alias="embed-model",
        input_text="{{text}}",
        normalize=True,
    )

    generator = EmbeddingColumnGenerator(
        config=config,
        resource_provider=mock_resource_provider,
    )

    # Setup mock response with non-normalized vector
    embedding_vector = [3.0, 4.0]  # Length = 5.0
    mock_response = EmbeddingResponse(
        data=[{"embedding": embedding_vector, "index": 0, "object": "embedding"}],
        model="test-model",
        usage=Usage(prompt_tokens=5, total_tokens=5),
    )
    mock_model_facade.embedding.return_value = mock_response

    # Generate embedding
    data = {"text": "Test text"}
    result = generator.generate(data)

    # Verify normalization
    normalized = result["normalized_embedding"]
    norm = np.linalg.norm(normalized)
    assert np.isclose(norm, 1.0, atol=1e-6)

    # Verify direction is preserved
    expected_normalized = [0.6, 0.8]  # [3/5, 4/5]
    assert np.allclose(normalized, expected_normalized, atol=1e-6)


def test_embedding_generator_with_complex_template(mock_resource_provider, mock_model_facade):
    """Test embedding generation with complex Jinja2 template."""
    config = EmbeddingColumnConfig(
        name="combined_embedding",
        model_alias="embed-model",
        input_text="{{title}} - {{description}}",
    )

    generator = EmbeddingColumnGenerator(
        config=config,
        resource_provider=mock_resource_provider,
    )

    # Setup mock response
    embedding_vector = [0.1, 0.2, 0.3]
    mock_response = EmbeddingResponse(
        data=[{"embedding": embedding_vector, "index": 0, "object": "embedding"}],
        model="test-model",
        usage=Usage(prompt_tokens=10, total_tokens=10),
    )
    mock_model_facade.embedding.return_value = mock_response

    # Generate embedding
    data = {"title": "Product", "description": "A great product"}
    result = generator.generate(data)

    # Verify
    assert result["combined_embedding"] == embedding_vector

    # Verify the input text was properly rendered
    call_args = mock_model_facade.embedding.call_args
    assert "Product - A great product" in str(call_args)


def test_embedding_generator_with_static_text(mock_resource_provider, mock_model_facade):
    """Test embedding generation with static text (no column references)."""
    config = EmbeddingColumnConfig(
        name="static_embedding",
        model_alias="embed-model",
        input_text="This is static text",
    )

    generator = EmbeddingColumnGenerator(
        config=config,
        resource_provider=mock_resource_provider,
    )

    # Setup mock response
    embedding_vector = [0.5, 0.5]
    mock_response = EmbeddingResponse(
        data=[{"embedding": embedding_vector, "index": 0, "object": "embedding"}],
        model="test-model",
        usage=Usage(prompt_tokens=5, total_tokens=5),
    )
    mock_model_facade.embedding.return_value = mock_response

    # Generate embedding
    data = {}
    result = generator.generate(data)

    # Verify
    assert result["static_embedding"] == embedding_vector


def test_embedding_generator_preserves_existing_data(embedding_generator, mock_model_facade):
    """Test that embedding generator preserves existing data in the record."""
    # Setup mock response
    embedding_vector = [0.1, 0.2]
    mock_response = EmbeddingResponse(
        data=[{"embedding": embedding_vector, "index": 0, "object": "embedding"}],
        model="test-model",
        usage=Usage(prompt_tokens=5, total_tokens=5),
    )
    mock_model_facade.embedding.return_value = mock_response

    # Generate embedding
    data = {"product_name": "Widget", "price": 99.99, "category": "Tools"}
    result = embedding_generator.generate(data)

    # Verify original data is preserved
    assert result["product_name"] == "Widget"
    assert result["price"] == 99.99
    assert result["category"] == "Tools"
    # And new embedding is added
    assert result["text_embedding"] == embedding_vector


def test_embedding_generator_zero_vector_normalization(mock_resource_provider, mock_model_facade):
    """Test that zero vectors are not normalized (to avoid division by zero)."""
    config = EmbeddingColumnConfig(
        name="zero_embedding",
        model_alias="embed-model",
        input_text="{{text}}",
        normalize=True,
    )

    generator = EmbeddingColumnGenerator(
        config=config,
        resource_provider=mock_resource_provider,
    )

    # Setup mock response with zero vector
    embedding_vector = [0.0, 0.0, 0.0]
    mock_response = EmbeddingResponse(
        data=[{"embedding": embedding_vector, "index": 0, "object": "embedding"}],
        model="test-model",
        usage=Usage(prompt_tokens=5, total_tokens=5),
    )
    mock_model_facade.embedding.return_value = mock_response

    # Generate embedding
    data = {"text": "Test"}
    result = generator.generate(data)

    # Verify zero vector is not normalized (would cause division by zero)
    assert result["zero_embedding"] == [0.0, 0.0, 0.0]


def test_embedding_generator_high_dimensional_vector(embedding_generator, mock_model_facade):
    """Test embedding generation with high-dimensional vectors."""
    # Setup mock response with 1536-dimensional vector (common for embeddings)
    embedding_vector = [float(i) / 1536 for i in range(1536)]
    mock_response = EmbeddingResponse(
        data=[{"embedding": embedding_vector, "index": 0, "object": "embedding"}],
        model="test-model",
        usage=Usage(prompt_tokens=5, total_tokens=5),
    )
    mock_model_facade.embedding.return_value = mock_response

    # Generate embedding
    data = {"product_name": "Widget"}
    result = embedding_generator.generate(data)

    # Verify
    assert len(result["text_embedding"]) == 1536
    assert result["text_embedding"] == embedding_vector
