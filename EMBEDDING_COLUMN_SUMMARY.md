# Embedding Column Implementation Summary

## Overview

Successfully implemented `EmbeddingColumnConfig` and `EmbeddingColumnGenerator` to enable automatic generation of embedding vector columns in datasets. This feature integrates seamlessly with the new `ModelFacade.embedding()` method and follows all existing DataDesigner patterns.

## Motivation

Embedding columns allow users to:
- Generate vector representations of text data for semantic search
- Create embeddings during dataset generation workflow
- Support downstream machine learning and similarity tasks
- Build RAG (Retrieval-Augmented Generation) datasets with embeddings

## Implementation Details

### 1. EmbeddingColumnConfig (`src/data_designer/config/column_configs.py`)

Added new column configuration type:

```python
class EmbeddingColumnConfig(SingleColumnConfig):
    input_text: str          # Jinja2 template for text to embed
    model_alias: str         # Embedding model to use
    normalize: bool = False  # Normalize vectors to unit length
    column_type: Literal["embedding"] = "embedding"
```

**Features:**
- Jinja2 templating support for combining multiple columns
- Validation of Jinja2 syntax at config time
- Automatic dependency tracking via `required_columns`
- Optional vector normalization for easier similarity comparisons

### 2. EmbeddingColumnGenerator (`src/data_designer/engine/column_generators/generators/llm_generators.py`)

Created generator that:
- Renders Jinja2 templates with row data
- Calls `ModelFacade.embedding()` method
- Optionally normalizes vectors using NumPy
- Handles zero vectors gracefully (no normalization to avoid division by zero)
- Preserves existing row data

```python
class EmbeddingColumnGenerator(ColumnGenerator[EmbeddingColumnConfig]):
    def generate(self, data: dict) -> dict:
        # Render input text template
        input_text = self.prompt_renderer.render(...)

        # Get embedding from model
        response = self.model.embedding(input_text, ...)
        embedding_vector = response.data[0]["embedding"]

        # Optionally normalize
        if self.config.normalize:
            embedding_vector = normalize(embedding_vector)

        data[self.config.name] = embedding_vector
        return data
```

### 3. Column Type Integration (`src/data_designer/config/column_types.py`)

Updated to include embedding column type:
- Added `EmbeddingColumnConfig` to `ColumnConfigT` union
- Added to `COLUMN_TYPE_EMOJI_MAP` with 🧬 emoji
- Included in `column_type_used_in_execution_dag()`
- Included in `column_type_is_llm_generated()`
- Added to display order and config factory

### 4. Registry Integration (`src/data_designer/engine/column_generators/registry.py`)

Registered the embedding column generator:

```python
registry.register(
    DataDesignerColumnType.EMBEDDING,
    EmbeddingColumnGenerator,
    EmbeddingColumnConfig
)
```

### 5. Public API (`src/data_designer/essentials/__init__.py`)

Exported for user access:
```python
from data_designer.essentials import EmbeddingColumnConfig
```

## Test Coverage

### Config Tests (`tests/config/test_embedding_column_config.py`)

- ✅ Basic configuration creation
- ✅ Normalization flag
- ✅ Required columns detection (single and multiple)
- ✅ Static text (no column references)
- ✅ Invalid Jinja2 template rejection
- ✅ Complex Jinja2 templates with filters
- ✅ Drop column functionality
- ✅ Side effect columns (none expected)
- ✅ Serialization/deserialization

### Generator Tests (`tests/engine/column_generators/test_embedding_generator.py`)

- ✅ Metadata verification
- ✅ Basic embedding generation
- ✅ Vector normalization
- ✅ Complex Jinja2 template rendering
- ✅ Static text embeddings
- ✅ Data preservation
- ✅ Zero vector handling (no normalization)
- ✅ High-dimensional vectors (1536-dim)

## Documentation

### Created Documentation Files

1. **`docs/columns/embedding-columns.md`**: Comprehensive guide covering:
   - Basic usage and configuration
   - 5 practical examples (product embeddings, combined text, semantic search, multilingual, intermediate embeddings)
   - Working with embeddings (similarity search, clustering, visualization)
   - Best practices
   - Advanced usage patterns
   - Performance considerations
   - Troubleshooting

2. **Updated `docs/models/advanced-model-usage.md`**: Now includes embedding functionality at the ModelFacade level

## Usage Examples

### Example 1: Basic Product Embeddings

```python
from data_designer.essentials import (
    DataDesignerConfigBuilder,
    EmbeddingColumnConfig,
    SamplerColumnConfig,
    CategorySamplerParams,
    SamplerType,
)

config_builder = DataDesignerConfigBuilder()

# Add product names
config_builder.add_column(
    SamplerColumnConfig(
        name="product_name",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["Laptop", "Mouse", "Keyboard"]
        ),
    )
)

# Generate embeddings
config_builder.add_column(
    EmbeddingColumnConfig(
        name="product_embedding",
        model_alias="embedder",
        input_text="{{ product_name }}",
        normalize=True,
    )
)

data_designer = DataDesigner(config_builder.build())
df = data_designer.generate(num_rows=100)
```

### Example 2: Combined Text Embeddings

```python
# Multiple text columns
config_builder.add_column(
    LLMTextColumnConfig(
        name="title",
        model_alias="text-model",
        prompt="Generate a product title for {{ category }}",
    )
)

config_builder.add_column(
    LLMTextColumnConfig(
        name="description",
        model_alias="text-model",
        prompt="Write a description for {{ title }}",
    )
)

# Combine for embedding
config_builder.add_column(
    EmbeddingColumnConfig(
        name="combined_embedding",
        model_alias="embedder",
        input_text="{{ title }}: {{ description }}",
        normalize=True,
    )
)
```

### Example 3: Semantic Search Dataset

```python
# Generate Q&A pairs with embeddings for semantic search
config_builder.add_column(
    LLMTextColumnConfig(
        name="question",
        model_alias="text-model",
        prompt="Generate a technical question",
    )
)

config_builder.add_column(
    LLMTextColumnConfig(
        name="answer",
        model_alias="text-model",
        prompt="Answer: {{ question }}",
    )
)

config_builder.add_column(
    EmbeddingColumnConfig(
        name="question_embedding",
        model_alias="embedder",
        input_text="{{ question }}",
        normalize=True,
    )
)
```

## Architecture Benefits

### 1. Consistency with Existing Patterns

- Follows same structure as `LLMTextColumnConfig`, `LLMCodeColumnConfig`, etc.
- Uses `WithLLMGeneration` patterns where applicable
- Integrates with existing DAG-based execution
- Respects column dependencies

### 2. Flexibility

- Jinja2 templating allows combining multiple columns
- Optional normalization for different use cases
- Compatible with any embedding model via LiteLLM
- Can be marked as intermediate column with `drop=True`

### 3. Type Safety

- Full type annotations throughout
- Pydantic validation for configuration
- LiteLLM's `EmbeddingResponse` types

### 4. Ease of Use

- Simple API matching existing column types
- Automatic dependency resolution
- Clear error messages from Jinja2 validation

## Files Modified

1. `src/data_designer/config/column_configs.py` - Added `EmbeddingColumnConfig`
2. `src/data_designer/config/column_types.py` - Integrated embedding column type
3. `src/data_designer/engine/column_generators/generators/llm_generators.py` - Added `EmbeddingColumnGenerator`
4. `src/data_designer/engine/column_generators/registry.py` - Registered embedding generator
5. `src/data_designer/essentials/__init__.py` - Exported `EmbeddingColumnConfig`

## Files Created

1. `tests/config/test_embedding_column_config.py` - Config tests (11 tests)
2. `tests/engine/column_generators/test_embedding_generator.py` - Generator tests (11 tests)
3. `docs/columns/embedding-columns.md` - Comprehensive documentation

## Integration with Previous Work

This feature builds on the `ModelFacade.embedding()` method implemented earlier:

1. **ModelFacade Enhancement**: Added `embedding()` method to support embedding API calls
2. **Embedding Column**: Now uses that method to generate embeddings in dataset workflows
3. **Complete Pipeline**: Users can now generate embeddings both:
   - Directly via `ModelFacade.embedding()` for ad-hoc usage
   - Automatically via `EmbeddingColumnConfig` in dataset generation

## Key Features

✅ **Jinja2 Templating** - Combine multiple columns into embedding input
✅ **Vector Normalization** - Optional L2 normalization for similarity tasks
✅ **Dependency Tracking** - Automatic detection of required columns
✅ **Type Safety** - Full type annotations and Pydantic validation
✅ **Test Coverage** - 22 comprehensive tests
✅ **Documentation** - Complete user guide with examples
✅ **Consistency** - Follows all existing DataDesigner patterns
✅ **Flexibility** - Works with any embedding model via LiteLLM

## Performance Characteristics

- **Generation Strategy**: Cell-by-cell (row-by-row)
- **API Calls**: One embedding API call per row
- **Memory**: Stores full embedding vectors in memory (typically 1536 dimensions)
- **Normalization**: O(n) where n is embedding dimension (negligible overhead)

## Future Enhancements (Optional)

### Potential Additions

1. **Batch Embedding**: Process multiple rows in single API call
   ```python
   EmbeddingColumnConfig(
       batch_size=100,  # Process 100 rows at once
   )
   ```

2. **Embedding Caching**: Cache embeddings for repeated text
   ```python
   EmbeddingColumnConfig(
       cache_embeddings=True,
   )
   ```

3. **Dimension Specification**: Allow custom dimensions
   ```python
   EmbeddingColumnConfig(
       dimensions=512,  # Request specific dimension size
   )
   ```

4. **Pooling Strategies**: Support different pooling for long text
   ```python
   EmbeddingColumnConfig(
       pooling="mean",  # mean, max, cls
   )
   ```

5. **Async Generation**: Async embedding generation
   ```python
   await generator.generate_async(data)
   ```

## Backward Compatibility

All changes are backward compatible:
- No breaking changes to existing APIs
- New column type is opt-in
- Existing column types unchanged
- No impact on existing datasets

## Conclusion

The `EmbeddingColumnConfig` and `EmbeddingColumnGenerator` implementation provides:

✅ **Complete Feature** - Config, generator, tests, docs
✅ **Production Ready** - Follows all best practices
✅ **Well Tested** - Comprehensive test coverage
✅ **Well Documented** - Complete user guide
✅ **Consistent** - Matches existing patterns
✅ **Extensible** - Easy to enhance in future

This feature enables users to generate embeddings as part of their synthetic dataset workflows, enabling powerful use cases like semantic search, RAG datasets, and similarity-based applications.

## Related Work

This implementation complements the earlier `ModelFacade` expansion:
- `ModelFacade.text_completion()` - Raw completions
- `ModelFacade.embedding()` - Embedding generation (used by this feature)
- `ModelType` enum - Model capability classification

Together, these provide a complete foundation for various model interaction patterns within DataDesigner.
