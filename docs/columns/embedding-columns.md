# Embedding Columns

Embedding columns generate high-dimensional vector representations of text inputs using embedding models. These vectors are useful for semantic search, similarity comparisons, clustering, and other vector-based operations.

## Overview

The `EmbeddingColumnConfig` allows you to automatically generate embeddings for text data in your dataset. This is particularly useful for:

- **Semantic Search**: Find similar items based on meaning rather than exact text matches
- **Clustering**: Group similar items together
- **Recommendations**: Find related content
- **Classification**: Use embeddings as features for downstream models
- **Similarity Matching**: Compare items semantically

## Basic Usage

```python
from data_designer.essentials import (
    DataDesignerConfigBuilder,
    EmbeddingColumnConfig,
    ModelConfig,
    ModelType,
    InferenceParameters,
)

# Configure an embedding model
embedding_model = ModelConfig(
    alias="embedder",
    model="nvidia/nv-embed-v2",
    inference_parameters=InferenceParameters(),
    model_type=ModelType.EMBEDDING,
)

config_builder = DataDesignerConfigBuilder(model_configs=[embedding_model])

# Add a column with text to embed
config_builder.add_column(
    SamplerColumnConfig(
        name="product_name",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["Laptop", "Mouse", "Keyboard", "Monitor"]
        ),
    )
)

# Add embedding column
config_builder.add_column(
    EmbeddingColumnConfig(
        name="product_embedding",
        model_alias="embedder",
        input_text="{{ product_name }}",
    )
)

# Build and generate
data_designer = DataDesigner(config_builder.build())
result = data_designer.generate(num_rows=100)
```

## Configuration Options

### EmbeddingColumnConfig

```python
class EmbeddingColumnConfig:
    name: str                    # Name of the embedding column
    input_text: str              # Jinja2 template for text to embed
    model_alias: str             # Alias of the embedding model
    normalize: bool = False      # Whether to normalize vectors to unit length
    drop: bool = False           # Whether to drop this column from final output
```

### Key Parameters

- **`input_text`**: Jinja2 template that will be rendered with the current row's data. Can reference other columns.
- **`model_alias`**: Must match a model configured with your `DataDesignerConfigBuilder`.
- **`normalize`**: If True, normalizes embedding vectors to unit length (useful for cosine similarity).

## Examples

### Example 1: Simple Product Embeddings

```python
from data_designer.essentials import (
    DataDesignerConfigBuilder,
    EmbeddingColumnConfig,
    SamplerColumnConfig,
    CategorySamplerParams,
    SamplerType,
)

config_builder = DataDesignerConfigBuilder()

# Product names
config_builder.add_column(
    SamplerColumnConfig(
        name="product",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["Gaming Mouse", "Wireless Keyboard", "4K Monitor"]
        ),
    )
)

# Generate embeddings
config_builder.add_column(
    EmbeddingColumnConfig(
        name="product_vector",
        model_alias="nvidia-embedding",  # Assumes you have this configured
        input_text="{{ product }}",
        normalize=True,  # Normalize for easier similarity comparisons
    )
)

data_designer = DataDesigner(config_builder.build())
df = data_designer.generate(num_rows=100)
```

### Example 2: Combined Text Embeddings

Combine multiple columns into a single embedding:

```python
# Multiple text columns
config_builder.add_column(
    SamplerColumnConfig(
        name="title",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(values=["AI Assistant", "Code Helper", "Data Tool"]),
    )
)

config_builder.add_column(
    SamplerColumnConfig(
        name="description",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["Helps with tasks", "Assists coding", "Analyzes data"]
        ),
    )
)

# Combine fields for embedding
config_builder.add_column(
    EmbeddingColumnConfig(
        name="combined_embedding",
        model_alias="embedder",
        input_text="{{ title }}: {{ description }}",
        normalize=True,
    )
)
```

### Example 3: Semantic Search System

Generate a dataset with embeddings for semantic search:

```python
from data_designer.essentials import (
    DataDesignerConfigBuilder,
    LLMTextColumnConfig,
    EmbeddingColumnConfig,
)

config_builder = DataDesignerConfigBuilder()

# Generate questions
config_builder.add_column(
    LLMTextColumnConfig(
        name="question",
        model_alias="nvidia-text",
        prompt="Generate a unique technical question about {{ topic }}",
    )
)

# Generate answers
config_builder.add_column(
    LLMTextColumnConfig(
        name="answer",
        model_alias="nvidia-text",
        prompt="Provide a detailed answer to: {{ question }}",
    )
)

# Embed the question for search
config_builder.add_column(
    EmbeddingColumnConfig(
        name="question_embedding",
        model_alias="embedder",
        input_text="{{ question }}",
        normalize=True,
    )
)

# Embed the answer for retrieval
config_builder.add_column(
    EmbeddingColumnConfig(
        name="answer_embedding",
        model_alias="embedder",
        input_text="{{ answer }}",
        normalize=True,
    )
)

data_designer = DataDesigner(config_builder.build())
qa_dataset = data_designer.generate(num_rows=1000)
```

### Example 4: Multilingual Embeddings

Generate embeddings for multilingual content:

```python
config_builder.add_column(
    SamplerColumnConfig(
        name="language",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(values=["en", "es", "fr", "de"]),
    )
)

config_builder.add_column(
    LLMTextColumnConfig(
        name="content",
        model_alias="multilingual-text",
        prompt="Write a short product description in language code: {{ language }}",
    )
)

config_builder.add_column(
    EmbeddingColumnConfig(
        name="multilingual_embedding",
        model_alias="multilingual-embedder",  # Use a multilingual embedding model
        input_text="{{ content }}",
        normalize=True,
    )
)
```

### Example 5: Intermediate Embeddings (Drop After Use)

Generate embeddings for intermediate processing without including them in final output:

```python
# Generate embedding but don't include in final dataset
config_builder.add_column(
    EmbeddingColumnConfig(
        name="temp_embedding",
        model_alias="embedder",
        input_text="{{ text }}",
        drop=True,  # This column will be dropped from final output
    )
)

# Use the embedding in a downstream column (e.g., for similarity calculation)
config_builder.add_column(
    ExpressionColumnConfig(
        name="similarity_score",
        expr="calculate_similarity(temp_embedding, reference_vector)",
    )
)
```

## Working with Embeddings

### Similarity Search

Once you have embeddings, you can perform similarity search:

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Get embeddings from generated dataset
embeddings = np.array(df["product_embedding"].tolist())

# Query embedding (from new product)
query_embedding = np.array([...])  # Get from embedding model

# Calculate similarities
similarities = cosine_similarity([query_embedding], embeddings)[0]

# Get top 5 most similar
top_indices = similarities.argsort()[-5:][::-1]
similar_products = df.iloc[top_indices]
```

### Clustering

Group similar items using embeddings:

```python
from sklearn.cluster import KMeans

# Get embeddings
embeddings = np.array(df["content_embedding"].tolist())

# Cluster into 5 groups
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Add cluster labels to dataframe
df["cluster"] = clusters
```

### Dimensionality Reduction for Visualization

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Get embeddings
embeddings = np.array(df["text_embedding"].tolist())

# Reduce to 2D for visualization
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
plt.title("Embedding Visualization")
plt.show()
```

## Best Practices

### 1. Choose the Right Embedding Model

- **General Text**: Use models like `nv-embed-v2` or `text-embedding-ada-002`
- **Code**: Use code-specific embedding models
- **Domain-Specific**: Use models fine-tuned for your domain (medical, legal, etc.)

### 2. Normalize When Needed

Use `normalize=True` when:
- Computing cosine similarity
- Comparing embeddings from different sources
- Building vector databases that use cosine distance

```python
EmbeddingColumnConfig(
    name="normalized_embedding",
    model_alias="embedder",
    input_text="{{ text }}",
    normalize=True,  # Good for similarity comparisons
)
```

### 3. Optimize Input Text

Craft meaningful input text for better embeddings:

```python
# Good: Provides context
input_text="Product: {{ name }}. Category: {{ category }}. Price: ${{ price }}"

# Less optimal: Missing context
input_text="{{ name }}"
```

### 4. Batch Generation

For large datasets, embeddings are generated row-by-row. Consider:
- Using appropriate batch sizes in your model configuration
- Monitoring API rate limits
- Caching embeddings when reusing the same text

### 5. Store Embeddings Efficiently

Embeddings are typically high-dimensional (e.g., 1536 dimensions):
- Use appropriate storage formats (numpy arrays, parquet with array columns)
- Consider quantization for large-scale applications
- Use vector databases for production similarity search

## Advanced Usage

### Custom Embedding Dimensions

Some models support custom embedding dimensions:

```python
from data_designer.essentials import InferenceParameters

embedding_model = ModelConfig(
    alias="custom-embedder",
    model="model-with-variable-dimensions",
    inference_parameters=InferenceParameters(
        extra_body={"dimensions": 512}  # Request 512-dim embeddings
    ),
    model_type=ModelType.EMBEDDING,
)
```

### Conditional Embeddings

Use conditional parameters for different embedding strategies:

```python
config_builder.add_column(
    EmbeddingColumnConfig(
        name="adaptive_embedding",
        model_alias="embedder",
        input_text="""
        {% if is_long_form %}
        {{ title }}: {{ full_content }}
        {% else %}
        {{ title }}
        {% endif %}
        """,
    )
)
```

## Performance Considerations

### API Costs

Embedding generation can incur API costs:
- Monitor usage through `usage_stats` on the model facade
- Consider caching embeddings for repeated text
- Use smaller dimension embeddings when possible

### Generation Speed

Embeddings are generated per row:
- For 1000 rows, expect 1000 API calls
- Consider parallel generation settings in `InferenceParameters`
- Balance speed vs. rate limits

```python
InferenceParameters(
    max_parallel_requests=10,  # Increase for faster generation
    timeout=30,
)
```

## Troubleshooting

### Issue: Embeddings are all zeros

**Solution**: Check that your embedding model is correctly configured and accessible.

### Issue: Normalized embeddings have varying lengths

**Solution**: Ensure `normalize=True` is set. Check for zero vectors in input.

### Issue: High memory usage

**Solution**: Generate in smaller batches or reduce embedding dimensions if supported by your model.

### Issue: Slow generation

**Solution**:
- Increase `max_parallel_requests` in `InferenceParameters`
- Check API rate limits
- Consider using a faster embedding model

## See Also

- [Advanced Model Usage](../models/advanced-model-usage.md) - Using embedding models directly
- [Model Configuration](../models/default-model-settings.md) - Configuring embedding models
- [Column Dependencies](../code_reference/column_configs.md) - Understanding column execution order
