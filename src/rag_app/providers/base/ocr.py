from abc import ABC, abstractmethod
from typing import Any


class OCRProvider(ABC):
    """
    Abstract interface for Optical Character Recognition (OCR) providers.
    Describes the capability to extract text regions and coordinates from document images.
    """

    @abstractmethod
    def extract_ocr(self, image_input: Any) -> list[Any]:
        """
        Extract OCR bounding boxes and text from image.
        """
        raise NotImplementedError
