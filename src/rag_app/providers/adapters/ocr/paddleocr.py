from typing import Any
from rag_app.providers.base.ocr import OCRProvider


class PaddleOCRAdapter(OCRProvider):
    """
    Adapter wrapping PaddleOCREngine for OCR extraction.
    """

    def __init__(self, **kwargs: Any):
        self._engine = None
        self._kwargs = kwargs

    def _get_engine(self):
        if self._engine is None:
            try:
                from directory_extractor.ocr.paddleocr_engine import PaddleOCREngine
                self._engine = PaddleOCREngine(**self._kwargs)
            except ImportError:
                # Fallback if directory_extractor optional OCR dependencies not installed
                self._engine = None
        return self._engine

    def extract_ocr(self, image_input: Any) -> list[Any]:
        engine = self._get_engine()
        if engine is None:
            return []
        return engine.extract(image_input)
