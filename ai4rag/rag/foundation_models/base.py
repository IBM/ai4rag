from abc import ABC, abstractmethod

from typing import Any


class FoundationModel(ABC):
    def __init__(self, model_id: str, model_params: dict[str, Any]):
        self.model_id = model_id
        self.model_params = model_params

    def __repr__(self) -> str:
        return self.model_id

    def __str__(self) -> str:
        return repr(self)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FoundationModel):
            return NotImplemented

        return self.model_id == other.model_id

    def __hash__(self):
        return hash(self.model_id)

    @abstractmethod
    def chat(self, system_message: str, user_message: str) -> str:
        """Docstring here"""
