# Advanced Model Usage

This guide covers advanced features of the `ModelFacade` class, including embeddings, raw text completions, and different model types.

## Overview

The `ModelFacade` class has been expanded beyond chat completions to support multiple model interaction patterns:

- **Chat Completions** (default): Conversational model interactions with message history
- **Text Completions**: Raw, non-chat completions for simpler use cases
- **Embeddings**: Generate vector embeddings for text inputs
- **Vision**: Multi-modal capabilities with image inputs (already supported via `multi_modal_context`)

## Model Types

Models can be categorized by their primary capability using the `ModelType` enum:

```python
from data_designer.config.models import ModelType

# Available model types
ModelType.CHAT        # For chat-based interactions (default)
ModelType.COMPLETION  # For raw text completions
ModelType.EMBEDDING   # For generating embeddings
ModelType.VISION      # For vision-capable models
```

### Configuring Model Types

When creating a `ModelConfig`, you can specify the model type:

```python
from data_designer.config.models import ModelConfig, ModelType, InferenceParameters

# Chat model (default)
chat_model = ModelConfig(
    alias="chat-model",
    model="nvidia/nvidia-nemotron-nano-9b-v2",
    inference_parameters=InferenceParameters(temperature=0.85),
    model_type=ModelType.CHAT  # This is the default
)

# Embedding model
embedding_model = ModelConfig(
    alias="embed-model",
    model="nvidia/nv-embed-v2",
    inference_parameters=InferenceParameters(),
    model_type=ModelType.EMBEDDING
)

# Completion model
completion_model = ModelConfig(
    alias="completion-model",
    model="nvidia/nvidia-nemotron-nano-9b-v2",
    inference_parameters=InferenceParameters(temperature=0.7),
    model_type=ModelType.COMPLETION
)
```

## Using Text Completions

Raw text completions provide a simpler interface for non-conversational generation tasks.

### Basic Usage

```python
from data_designer.engine.models.facade import ModelFacade
from data_designer.config.models import ModelConfig, InferenceParameters

# Create a model facade
model_config = ModelConfig(
    alias="completion-model",
    model="nvidia/nvidia-nemotron-nano-9b-v2",
    inference_parameters=InferenceParameters(temperature=0.7),
)

facade = ModelFacade(
    model_config=model_config,
    secret_resolver=secret_resolver,
    model_provider_registry=provider_registry,
)

# Generate a text completion
prompt = "Complete this sentence: The future of AI is"
response = facade.text_completion(prompt)

# Access the completion
completed_text = response.choices[0].text
print(completed_text)
```

### Advanced Options

```python
# With generation parameters
response = facade.text_completion(
    prompt="Explain quantum computing in simple terms:",
    temperature=0.5,
    max_tokens=200,
    top_p=0.9
)

# Skip usage tracking (useful for testing)
response = facade.text_completion(
    prompt="Test prompt",
    skip_usage_tracking=True
)
```

## Using Embeddings

Generate vector embeddings for text inputs, useful for semantic search, clustering, and similarity comparisons.

### Single Text Embedding

```python
from data_designer.engine.models.facade import ModelFacade
from data_designer.config.models import ModelConfig, ModelType, InferenceParameters

# Create an embedding model facade
model_config = ModelConfig(
    alias="embed-model",
    model="nvidia/nv-embed-v2",
    inference_parameters=InferenceParameters(),
    model_type=ModelType.EMBEDDING
)

facade = ModelFacade(
    model_config=model_config,
    secret_resolver=secret_resolver,
    model_provider_registry=provider_registry,
)

# Generate embedding for a single text
text = "Machine learning is transforming technology"
response = facade.embedding(text)

# Access the embedding vector
embedding_vector = response.data[0]["embedding"]
print(f"Embedding dimension: {len(embedding_vector)}")
```

### Batch Embeddings

Process multiple texts in a single request for efficiency:

```python
# Generate embeddings for multiple texts
texts = [
    "First document about AI",
    "Second document about machine learning",
    "Third document about deep learning",
]

response = facade.embedding(texts)

# Access all embeddings
for i, item in enumerate(response.data):
    embedding = item["embedding"]
    print(f"Text {i}: dimension={len(embedding)}")
```

### Embedding Use Cases

#### Semantic Search

```python
import numpy as np

# Embed query and documents
query = "What is artificial intelligence?"
documents = [
    "AI is the simulation of human intelligence",
    "Machine learning is a subset of AI",
    "Natural language processing enables AI to understand text",
]

query_response = facade.embedding(query)
query_vector = np.array(query_response.data[0]["embedding"])

docs_response = facade.embedding(documents)
doc_vectors = [np.array(item["embedding"]) for item in docs_response.data]

# Calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Find most similar document
similarities = [cosine_similarity(query_vector, doc) for doc in doc_vectors]
most_similar_idx = np.argmax(similarities)
print(f"Most similar document: {documents[most_similar_idx]}")
```

#### Text Clustering

```python
from sklearn.cluster import KMeans

# Generate embeddings for a corpus
corpus = [
    "AI and machine learning",
    "Deep learning networks",
    "Quantum computing basics",
    "Quantum algorithms",
    "Neural networks",
]

response = facade.embedding(corpus)
embeddings = np.array([item["embedding"] for item in response.data])

# Cluster the embeddings
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Group texts by cluster
for cluster_id in range(2):
    cluster_texts = [corpus[i] for i, c in enumerate(clusters) if c == cluster_id]
    print(f"Cluster {cluster_id}: {cluster_texts}")
```

## Model Usage Tracking

All methods (`completion`, `text_completion`, `embedding`) support automatic usage tracking:

```python
# Check usage statistics
print(facade.usage_stats.model_dump())

# Output example:
# {
#     "token_usage": {
#         "prompt_tokens": 150,
#         "completion_tokens": 75,
#         "total_tokens": 225
#     },
#     "request_usage": {
#         "successful_requests": 5,
#         "failed_requests": 0,
#         "total_requests": 5
#     }
# }

# Skip tracking for specific calls
response = facade.embedding("test", skip_usage_tracking=True)
```

## Vision Models (Multi-Modal)

Vision capabilities are already supported through the existing `multi_modal_context` parameter in chat completions. See the [Multi-Modal section](../notebooks/1-the-basics.ipynb) for more details.

```python
from data_designer.config.models import ImageContext, ModalityDataType, ImageFormat

# Configure image context
image_context = ImageContext(
    column_name="product_image",
    data_type=ModalityDataType.BASE64,
    image_format=ImageFormat.PNG
)

# Use in generation
response, reasoning = facade.generate(
    prompt="Describe this product image in detail",
    system_prompt="You are a product description specialist",
    parser=lambda x: x,
    multi_modal_context=[image_context.get_context({"product_image": base64_image})]
)
```

## Best Practices

### Choose the Right Method

- **Use `completion()`** for conversational interactions, multi-turn dialogues, and when you need system prompts
- **Use `text_completion()`** for simple, one-shot completions without conversational context
- **Use `embedding()`** for semantic search, clustering, similarity comparisons, and vector operations
- **Use `generate()`** when you need parsing, validation, and error correction

### Batch Processing

For embeddings, always batch multiple texts together when possible:

```python
# ❌ Inefficient: Multiple individual calls
embeddings = []
for text in texts:
    response = facade.embedding(text)
    embeddings.append(response.data[0]["embedding"])

# ✅ Efficient: Single batch call
response = facade.embedding(texts)
embeddings = [item["embedding"] for item in response.data]
```

### Error Handling

All methods may raise exceptions from the underlying LiteLLM router:

```python
try:
    response = facade.embedding("test input")
except Exception as e:
    print(f"Embedding generation failed: {e}")
    # Handle error appropriately
```

### Usage Tracking

Monitor usage statistics to track costs and performance:

```python
# Check stats before and after
before_tokens = facade.usage_stats.token_usage.total_tokens
response = facade.embedding(texts)
after_tokens = facade.usage_stats.token_usage.total_tokens

tokens_used = after_tokens - before_tokens
print(f"This request used {tokens_used} tokens")
```

## API Reference

### `ModelFacade.text_completion()`

```python
def text_completion(
    self,
    prompt: str,
    skip_usage_tracking: bool = False,
    **kwargs
) -> TextCompletionResponse
```

**Parameters:**
- `prompt` (str): The input text prompt for completion
- `skip_usage_tracking` (bool): Whether to skip tracking usage statistics
- `**kwargs`: Additional arguments (temperature, max_tokens, etc.)

**Returns:** `TextCompletionResponse` from LiteLLM

### `ModelFacade.embedding()`

```python
def embedding(
    self,
    input_text: str | list[str],
    skip_usage_tracking: bool = False,
    **kwargs
) -> EmbeddingResponse
```

**Parameters:**
- `input_text` (str | list[str]): Single string or list of strings to embed
- `skip_usage_tracking` (bool): Whether to skip tracking usage statistics
- `**kwargs`: Additional arguments (dimensions, etc.)

**Returns:** `EmbeddingResponse` from LiteLLM containing embedding vectors

### `ModelFacade.model_type`

```python
@property
def model_type(self) -> ModelType
```

**Returns:** The configured `ModelType` for this model

## Examples

### Complete Example: RAG System

Here's a complete example building a simple RAG (Retrieval-Augmented Generation) system:

```python
import numpy as np
from data_designer.config.models import (
    ModelConfig,
    ModelType,
    InferenceParameters,
)
from data_designer.engine.models.facade import ModelFacade

# Setup embedding model
embed_config = ModelConfig(
    alias="embed",
    model="nvidia/nv-embed-v2",
    inference_parameters=InferenceParameters(),
    model_type=ModelType.EMBEDDING,
)
embed_facade = ModelFacade(embed_config, secret_resolver, provider_registry)

# Setup chat model
chat_config = ModelConfig(
    alias="chat",
    model="nvidia/nvidia-nemotron-nano-9b-v2",
    inference_parameters=InferenceParameters(temperature=0.7),
    model_type=ModelType.CHAT,
)
chat_facade = ModelFacade(chat_config, secret_resolver, provider_registry)

# Knowledge base
knowledge_base = [
    "Python is a high-level programming language",
    "Machine learning enables computers to learn from data",
    "Neural networks are inspired by biological neurons",
]

# Index the knowledge base
kb_response = embed_facade.embedding(knowledge_base)
kb_embeddings = np.array([item["embedding"] for item in kb_response.data])

# User query
query = "What is Python?"
query_response = embed_facade.embedding(query)
query_embedding = np.array(query_response.data[0]["embedding"])

# Find most relevant document
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarities = [cosine_similarity(query_embedding, kb_emb) for kb_emb in kb_embeddings]
most_relevant_idx = np.argmax(similarities)
context = knowledge_base[most_relevant_idx]

# Generate answer with context
messages = [
    {"role": "system", "content": "Answer based on the provided context."},
    {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
]
response = chat_facade.completion(messages)
answer = response.choices[0].message.content

print(f"Answer: {answer}")
```

## See Also

- [Default Model Settings](default-model-settings.md)
- [Column Configurations](../code_reference/column_configs.md)
- [The Basics Notebook](../notebooks/1-the-basics.ipynb)
