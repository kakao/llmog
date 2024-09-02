from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class MinimalTemplate(ABC):
    @abstractmethod
    def __init__(self):
        self.name = None

    def get_name(self):
        return self.name

    @abstractmethod
    def get_answer_choices_list(self, example: Any) -> List[str]:
        return []

    @abstractmethod
    def apply(self, example: Any) -> Tuple[str]:
        return (None, None)
