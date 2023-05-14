from abc import abstractmethod, ABC

class BasePreprocessor(ABC):
    @abstractmethod
    async def ingest(self) -> None:
        pass

    @abstractmethod
    async def split_data(self) -> None:
        pass

    @abstractmethod
    async def preprocess(self) -> None:
        pass

    @abstractmethod
    async def preprocess_batch(self) -> None:
        pass

    @abstractmethod
    async def save_data(self) -> None:
        pass