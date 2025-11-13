# Default Model Settings

Data Designer ships with pre-configured model providers and model configurations that make it easy to start generating synthetic data without manual setup.

## Model Providers

Data Designer includes two default model providers that are automatically configured when you create a `DataDesigner` instance:

### NVIDIA Provider (`nvidia`)

- **Endpoint**: `https://integrate.api.nvidia.com/v1`
- **API Key**: Set via `NVIDIA_API_KEY` environment variable
- **Models**: Access to NVIDIA's hosted models from [build.nvidia.com](https://build.nvidia.com)
- **Getting Started**: Sign up and get your API key at [build.nvidia.com](https://build.nvidia.com)

The NVIDIA provider gives you access to state-of-the-art models including Nemotron and other NVIDIA-optimized models.

### OpenAI Provider (`openai`)

- **Endpoint**: `https://api.openai.com/v1`
- **API Key**: Set via `OPENAI_API_KEY` environment variable
- **Models**: Access to OpenAI's model catalog
- **Getting Started**: Get your API key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

The OpenAI provider gives you access to GPT models and other OpenAI offerings.

### How Default Providers Work

When you create a `DataDesigner` instance without specifying model providers, both providers are automatically configured if their respective API keys are set in your environment:

```python
from data_designer.essentials import DataDesigner

# Both providers are automatically configured based on available API keys
data_designer = DataDesigner()
```

## Model Configurations

Data Designer provides pre-configured model aliases for common use cases. When you create a `DataDesignerConfigBuilder` without specifying `model_configs`, these default configurations are automatically available.

### NVIDIA Models

The following model configurations are automatically available when `NVIDIA_API_KEY` is set:

| Alias | Model | Use Case | Temperature | Top P |
|-------|-------|----------|-------------|-------|
| `nvidia-text` | `nvidia/nvidia-nemotron-nano-9b-v2` | General text generation | 0.85 | 0.95 |
| `nvidia-reasoning` | `openai/gpt-oss-20b` | Reasoning and analysis tasks | 0.35 | 0.95 |
| `nvidia-vision` | `nvidia/nemotron-nano-12b-v2-vl` | Vision and image understanding | 0.85 | 0.95 |

**Usage Example:**

```python
from data_designer.essentials import (
    DataDesignerConfigBuilder,
    LLMTextColumnConfig,
)

config_builder = DataDesignerConfigBuilder()

# Use the pre-configured nvidia-text model
config_builder.add_column(
    LLMTextColumnConfig(
        name="description",
        model_alias="nvidia-text",
        prompt="Generate a product description",
    )
)
```

### OpenAI Models

The following model configurations are automatically available when `OPENAI_API_KEY` is set:

| Alias | Model | Use Case | Temperature | Top P |
|-------|-------|----------|-------------|-------|
| `openai-text` | `gpt-4.1` | General text generation | 0.85 | 0.95 |
| `openai-reasoning` | `gpt-5` | Reasoning and analysis tasks | 0.35 | 0.95 |
| `openai-vision` | `gpt-5` | Vision and image understanding | 0.85 | 0.95 |

**Usage Example:**

```python
from data_designer.essentials import (
    DataDesignerConfigBuilder,
    LLMTextColumnConfig,
)

config_builder = DataDesignerConfigBuilder()

# Use the pre-configured openai-reasoning model
config_builder.add_column(
    LLMTextColumnConfig(
        name="analysis",
        model_alias="openai-reasoning",
        prompt="Analyze the following data: {{data}}",
    )
)
```

## Important Notes

!!! warning "API Key Requirements"
    If neither `NVIDIA_API_KEY` nor `OPENAI_API_KEY` is set, you'll need to provide custom model configurations to use Data Designer. The default configurations will not be available without at least one API key.

!!! tip "Environment Variables"
    Store your API keys in environment variables rather than hardcoding them in your scripts:

    ```bash
    # In your .bashrc, .zshrc, or similar
    export NVIDIA_API_KEY="your-api-key-here"
    export OPENAI_API_KEY="your-openai-api-key-here"
    ```

## See Also

- **[Quick Start Guide](../quick-start.md)**: Get started with a simple example
- **[Model Configuration Reference](../code_reference/config_builder.md)**: Detailed API documentation
- **[Column Configurations](../code_reference/column_configs.md)**: Learn about all column types
