from abc import abstractmethod, ABC

class BasePreprocessor(ABC):
    @abstractmethod
    async def ingest(self) -> None:
        pass

    @abstractmethod
    async def split_data(self) -> None:
        pass

    @abstractmethod
    async def preprocess_text(self) -> None:
        pass

    @abstractmethod
    async def preprocess_text_batch(self) -> None:
        pass

    @abstractmethod
    async def preprocess_categorical(self) -> None:
        pass
    
    @abstractmethod
    async def preprocess_numerical(self) -> None:
        pass

    @abstractmethod
    async def save_data(self) -> None:
        pass