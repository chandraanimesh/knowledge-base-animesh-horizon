import hashlib
import re
from typing import List

from rag_app.research.provider import WebPageResult
from rag_app.schemas.research import CleanWebPage
from rag_app.ingestion.chunker import normalize_text

MAX_PAGE_CLEAN_CHARS = 16000


def clean_web_page(page: WebPageResult, max_chars: int = MAX_PAGE_CLEAN_CHARS) -> CleanWebPage:
    """
    Normalizes a WebPageResult into a CleanWebPage.
    Uses basic deterministic transformations to clean whitespace and noise,
    bounds initial character volume, and tracks truncation provenance.
    """
    meta = dict(page.metadata or {})
    if not page.markdown:
        return CleanWebPage(
            url=page.url,
            title=page.title,
            content="",
            metadata=meta,
        )
    
    cleaned_text = normalize_text(page.markdown)
    cleaned_text = re.sub(r'!\[.*?\]\(.*?\)', '', cleaned_text).strip()

    if len(cleaned_text) > max_chars:
        # Bounded initial character trimming at paragraph boundary
        truncated_slice = cleaned_text[:max_chars]
        last_para = truncated_slice.rfind("\n\n")
        if last_para > int(max_chars * 0.7):
            cleaned_text = truncated_slice[:last_para].strip()
        else:
            cleaned_text = truncated_slice.strip()
        meta["is_truncated"] = True
        meta["truncated_from_chars"] = len(page.markdown)
        meta["clean_chars"] = len(cleaned_text)

    return CleanWebPage(
        url=page.url,
        title=page.title,
        content=cleaned_text,
        metadata=meta,
    )


def clean_and_deduplicate(pages: List[WebPageResult]) -> List[CleanWebPage]:
    """
    Cleans a list of WebPageResult and removes duplicates based on content hash.
    """
    seen_hashes = set()
    clean_pages = []
    
    for page in pages:
        clean_page = clean_web_page(page)
        
        # Don't keep completely empty pages
        if not clean_page.content:
            continue
            
        content_hash = hashlib.sha256(clean_page.content.encode('utf-8')).hexdigest()
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            clean_pages.append(clean_page)
            
    return clean_pages
