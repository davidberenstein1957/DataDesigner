# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from data_designer.config.column_configs import (
    LLMCodeColumnConfig,
    LLMJudgeColumnConfig,
    LLMStructuredColumnConfig,
    LLMTextColumnConfig,
)
from data_designer.engine.column_generators.generators.base import (
    ColumnGenerator,
    GenerationStrategy,
    GeneratorMetadata,
)
from data_designer.engine.column_generators.generators.generation_mixins import (
    DEFAULT_MAX_CONVERSATION_RESTARTS,
    WithCompletionGeneration,
)
from data_designer.engine.resources.resource_provider import ResourceType

logger = logging.getLogger(__name__)


class LLMTextCellGenerator(WithCompletionGeneration, ColumnGenerator[LLMTextColumnConfig]):
    @staticmethod
    def metadata() -> GeneratorMetadata:
        return GeneratorMetadata(
            name="llm_text_generator",
            description="Generate a new dataset cell from a prompt template",
            generation_strategy=GenerationStrategy.CELL_BY_CELL,
            required_resources=[ResourceType.MODEL_REGISTRY],
        )


class LLMCodeCellGenerator(WithCompletionGeneration, ColumnGenerator[LLMCodeColumnConfig]):
    @staticmethod
    def metadata() -> GeneratorMetadata:
        return GeneratorMetadata(
            name="llm_code_generator",
            description="Generate a new dataset cell from a prompt template",
            generation_strategy=GenerationStrategy.CELL_BY_CELL,
            required_resources=[ResourceType.MODEL_REGISTRY],
        )


class LLMStructuredCellGenerator(WithCompletionGeneration, ColumnGenerator[LLMStructuredColumnConfig]):
    @staticmethod
    def metadata() -> GeneratorMetadata:
        return GeneratorMetadata(
            name="llm_structured_generator",
            description="Generate a new dataset cell from a prompt template",
            generation_strategy=GenerationStrategy.CELL_BY_CELL,
            required_resources=[ResourceType.MODEL_REGISTRY],
        )


class LLMJudgeCellGenerator(WithCompletionGeneration, ColumnGenerator[LLMJudgeColumnConfig]):
    @staticmethod
    def metadata() -> GeneratorMetadata:
        return GeneratorMetadata(
            name="llm_judge_generator",
            description="Judge a new dataset cell based on a set of rubrics",
            generation_strategy=GenerationStrategy.CELL_BY_CELL,
            required_resources=[ResourceType.MODEL_REGISTRY],
        )

    @property
    def max_conversation_restarts(self) -> int:
        return DEFAULT_MAX_CONVERSATION_RESTARTS * 2
