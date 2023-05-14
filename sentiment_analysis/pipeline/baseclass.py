from abc import abstractmethod, ABC

class BasePipeline(ABC):
    @abstractmethod
    async def preprocessing_pipeline(self) -> None:
        pass

    @abstractmethod
    async def model_pipeline(self) -> None:
        pass

    # @abstractmethod
    # async def inference_pipeline(self) -> None:
    #     pass