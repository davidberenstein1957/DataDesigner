# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import ValidationError
import pytest

from data_designer.config.column_configs import EmbeddingColumnConfig
from data_designer.config.errors import InvalidConfigError


def test_embedding_column_config_basic():
    """Test basic embedding column config creation."""
    config = EmbeddingColumnConfig(
        name="text_embedding",
        model_alias="embed-model",
        input_text="{{product_name}}",
    )

    assert config.name == "text_embedding"
    assert config.model_alias == "embed-model"
    assert config.input_text == "{{product_name}}"
    assert config.normalize is False
    assert config.column_type == "embedding"


def test_embedding_column_config_with_normalize():
    """Test embedding column config with normalization enabled."""
    config = EmbeddingColumnConfig(
        name="normalized_embedding",
        model_alias="embed-model",
        input_text="{{text}}",
        normalize=True,
    )

    assert config.normalize is True


def test_embedding_column_config_required_columns_single():
    """Test that required_columns correctly identifies single referenced column."""
    config = EmbeddingColumnConfig(
        name="embedding",
        model_alias="embed-model",
        input_text="{{description}}",
    )

    assert config.required_columns == ["description"]


def test_embedding_column_config_required_columns_multiple():
    """Test that required_columns correctly identifies multiple referenced columns."""
    config = EmbeddingColumnConfig(
        name="combined_embedding",
        model_alias="embed-model",
        input_text="{{title}} - {{description}} - {{category}}",
    )

    required = set(config.required_columns)
    assert required == {"title", "description", "category"}


def test_embedding_column_config_required_columns_none():
    """Test required_columns with static text (no column references)."""
    config = EmbeddingColumnConfig(
        name="static_embedding",
        model_alias="embed-model",
        input_text="This is static text",
    )

    assert config.required_columns == []


def test_embedding_column_config_invalid_jinja_template():
    """Test that invalid Jinja2 templates raise validation error."""
    with pytest.raises(InvalidConfigError, match="Invalid Jinja2 template"):
        EmbeddingColumnConfig(
            name="invalid_embedding",
            model_alias="embed-model",
            input_text="{{unclosed_template",
        )


def test_embedding_column_config_complex_jinja_template():
    """Test complex Jinja2 template with filters and control structures."""
    config = EmbeddingColumnConfig(
        name="complex_embedding",
        model_alias="embed-model",
        input_text="{% if title %}{{ title | upper }}{% endif %} - {{ description }}",
    )

    required = set(config.required_columns)
    assert "title" in required
    assert "description" in required


def test_embedding_column_config_drop_column():
    """Test embedding column with drop=True."""
    config = EmbeddingColumnConfig(
        name="temp_embedding",
        model_alias="embed-model",
        input_text="{{text}}",
        drop=True,
    )

    assert config.drop is True


def test_embedding_column_config_side_effect_columns():
    """Test that embedding columns have no side effect columns."""
    config = EmbeddingColumnConfig(
        name="embedding",
        model_alias="embed-model",
        input_text="{{text}}",
    )

    assert config.side_effect_columns == []


def test_embedding_column_config_missing_required_fields():
    """Test that missing required fields raise validation error."""
    with pytest.raises(ValidationError):
        EmbeddingColumnConfig(
            name="embedding",
            # Missing model_alias and input_text
        )


def test_embedding_column_config_serialization():
    """Test that embedding column config can be serialized and deserialized."""
    config = EmbeddingColumnConfig(
        name="embedding",
        model_alias="embed-model",
        input_text="{{title}} - {{description}}",
        normalize=True,
    )

    # Serialize to dict
    config_dict = config.model_dump()

    # Deserialize from dict
    restored_config = EmbeddingColumnConfig(**config_dict)

    assert restored_config.name == config.name
    assert restored_config.model_alias == config.model_alias
    assert restored_config.input_text == config.input_text
    assert restored_config.normalize == config.normalize
