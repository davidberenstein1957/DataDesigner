# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pandas as pd

from data_designer.config.columns import CustomColumnConfig
from data_designer.engine.column_generators.generators.base import (
    ColumnGenerator,
    GenerationStrategy,
    GeneratorMetadata,
)
from data_designer.engine.errors import DataDesignerRuntimeError


logger = logging.getLogger(__name__)


class CustomColumnGenerator(ColumnGenerator[CustomColumnConfig]):
    @staticmethod
    def metadata() -> GeneratorMetadata:
        return GeneratorMetadata(
            name="custom",
            description="Generate a custom column.",
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            required_resources=None,
        )

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"🛠️ Generating custom column {self.config.name!r} with {len(data)} records")
        logger.info(f"  |-- generator function: {self.config.generator_function.__name__}")

        try:
            result = self.config.generator_function(data)
        except Exception as e:
            raise DataDesignerRuntimeError(f"Error generating custom column {self.config.name!r}: {e}")

        return result