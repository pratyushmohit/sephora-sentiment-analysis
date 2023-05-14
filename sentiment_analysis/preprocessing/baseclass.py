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
    def preprocess_y(self) -> None:
        pass

    @abstractmethod
    async def tokenization(self) -> None:
        pass

    @abstractmethod
    async def glove_embedding(self) -> None:
        pass

    @abstractmethod
    async def padding(self) -> None:
        pass

    @abstractmethod
    async def save_data(self) -> None:
        pass