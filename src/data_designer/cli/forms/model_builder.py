# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import textwrap
from typing import Any

from data_designer.cli.forms.builder import FormBuilder
from data_designer.cli.forms.field import DictField, SelectField, TextField
from data_designer.cli.forms.form import Form
from data_designer.config.models import GenerationType, ModelConfig


class ModelFormBuilder(FormBuilder[ModelConfig]):
    """Builds interactive forms for model configuration."""

    def __init__(self, existing_aliases: set[str] | None = None, available_providers: list[str] | None = None):
        super().__init__("Model Configuration")
        self.existing_aliases = existing_aliases or set()
        self.available_providers = available_providers or []

    def create_form(self, initial_data: dict[str, Any] | None = None) -> Form:
        """Create the model configuration form."""
        fields = []

        # Model alias
        fields.append(
            TextField(
                "alias",
                "Model alias (used in your configs)",
                default=initial_data.get("alias") if initial_data else None,
                required=True,
                validator=self._validate_alias,
            )
        )

        # Model ID
        fields.append(
            TextField(
                "model",
                "Model ID",
                default=initial_data.get("model") if initial_data else None,
                required=True,
                validator=lambda x: (False, "Model ID is required") if not x else (True, None),
            )
        )

        # Provider (if multiple available)
        if len(self.available_providers) > 1:
            provider_options = {p: p for p in self.available_providers}
            fields.append(
                SelectField(
                    "provider",
                    "Select provider for this model",
                    options=provider_options,
                    default=initial_data.get("provider", self.available_providers[0])
                    if initial_data
                    else self.available_providers[0],
                )
            )
        elif len(self.available_providers) == 1:
            # Single provider - will be set automatically
            pass

        # Generation type
        fields.append(
            SelectField(
                "generation_type",
                "Generation type",
                options={
                    GenerationType.CHAT_COMPLETION: "Chat completion",
                    GenerationType.EMBEDDING: "Embedding",
                },
                default=initial_data.get("generation_type", GenerationType.CHAT_COMPLETION)
                if initial_data
                else GenerationType.CHAT_COMPLETION,
            )
        )

        # Inference parameters as dictionary
        default_inference_params = initial_data.get("inference_parameters") if initial_data else {}

        inference_params_instructions = textwrap.dedent("""
        Inference parameters
        |-- (Enter as JSON)
        |-- Hit enter to accept model defaults.
        |-- E.g., {"temperature": 0.7, "top_p": 0.9, "max_tokens": 2048} for chat completion models
        |-- E.g., {"encoding_format": "float", "dimensions": 1024} for embedding models
        """)
        fields.append(
            DictField(
                "inference_parameters",
                inference_params_instructions,
                default=default_inference_params,
                required=True,
            )
        )

        return Form(self.title, fields)

    def _validate_alias(self, alias: str) -> tuple[bool, str | None]:
        """Validate model alias."""
        if not alias:
            return False, "Model alias is required"
        if alias in self.existing_aliases:
            return False, f"Model alias '{alias}' already exists"
        return True, None

    def build_config(self, form_data: dict[str, Any]) -> ModelConfig:
        """Build ModelConfig from form data."""
        # Determine provider
        if "provider" in form_data:
            provider = form_data["provider"]
        elif len(self.available_providers) == 1:
            provider = self.available_providers[0]
        else:
            provider = None

        # Get generation type
        generation_type = form_data.get("generation_type", GenerationType.CHAT_COMPLETION)

        # Get inference parameters and add max_parallel_requests if not present
        inference_params = form_data.get("inference_parameters", {})
        if "max_parallel_requests" not in inference_params:
            inference_params["max_parallel_requests"] = 4

        return ModelConfig(
            alias=form_data["alias"],
            model=form_data["model"],
            provider=provider,
            generation_type=generation_type,
            inference_parameters=inference_params,
        )
