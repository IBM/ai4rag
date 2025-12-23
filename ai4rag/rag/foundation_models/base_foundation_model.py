# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Generic, TypeVar


FoundationModelClientT = TypeVar("FoundationModelClientT")
FoundationModelParamsT = TypeVar("FoundationModelParamsT")


class BaseFoundationModel(Generic[FoundationModelClientT, FoundationModelParamsT], ABC):
    def __init__(self, client: FoundationModelClientT, model_id: str, model_params: FoundationModelParamsT):
        self.client = client
        self.model_id = model_id
        self.model_params = model_params

    def __repr__(self) -> str:
        return self.model_id

    def __str__(self) -> str:
        return repr(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseFoundationModel):
            raise NotImplementedError

        return self.model_id == other.model_id

    def __hash__(self):
        return hash(self.model_id)

    @abstractmethod
    def chat(self, system_message: str, user_message: str) -> str:
        """Docstring here"""
