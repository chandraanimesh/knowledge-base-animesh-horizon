import asyncio
import logging
import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid

from rag_app.db.models import DocumentChunk
from rag_app.db.session import AsyncSessionLocal

logger = logging.getLogger(__name__)

class GlobalBM25Index:
    _instance = None
    
    def __init__(self):
        self.bm25 = None
        self.chunk_ids = []
        self._lock = asyncio.Lock()
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return [w for w in re.findall(r'\w+', text.lower()) if w]
        
    async def build(self, session: AsyncSession):
        """Build the index from all chunks in DB."""
        async with self._lock:
            logger.info("Building Global BM25 Index...")
            
            statement = select(
                DocumentChunk.id, 
                DocumentChunk.text, 
                DocumentChunk.metadata_
            )
            
            result = await session.execute(statement)
            rows = result.all()
            
            corpus = []
            self.chunk_ids = []
            
            for row in rows:
                chunk_id = row[0]
                text = row[1]
                metadata_ = row[2] or {}
                
                index_text = metadata_.get('embedding_text')
                if not index_text:
                    context_parts = []
                    if metadata_.get('search_context'):
                        context_parts.append(metadata_['search_context'])
                    else:
                        for k in sorted(metadata_.keys()):
                            if k.startswith('header_') and metadata_[k]:
                                context_parts.append(str(metadata_[k]))
                    if context_parts:
                        index_text = f"{'\n'.join(context_parts)}\n{text}"
                    else:
                        index_text = text
                
                tokens = self._tokenize(index_text)
                corpus.append(tokens)
                self.chunk_ids.append(chunk_id)
                
            if corpus:
                self.bm25 = BM25Okapi(corpus)
            else:
                self.bm25 = None
                
            logger.info(f"Global BM25 Index built with {len(self.chunk_ids)} chunks.")
            
    async def get_scores(self, query: str) -> Dict[uuid.UUID, float]:
        """Returns a dict mapping chunk_id to bm25 score"""
        async with self._lock:
            if not self.bm25:
                return {}
                
            tokens = self._tokenize(query)
            if not tokens:
                return {}
                
            scores = self.bm25.get_scores(tokens)
            
            return {chunk_id: float(score) for chunk_id, score in zip(self.chunk_ids, scores) if score > 0}
