import hashlib
import re
from typing import List

from rag_app.research.provider import WebPageResult
from rag_app.schemas.research import CleanWebPage
from rag_app.ingestion.chunker import normalize_text

def clean_web_page(page: WebPageResult) -> CleanWebPage:
    """
    Normalizes a WebPageResult into a CleanWebPage.
    Uses basic deterministic transformations to clean whitespace and noise,
    without using an LLM or altering factual content.
    """
    if not page.markdown:
        # Empty content
        return CleanWebPage(
            url=page.url,
            title=page.title,
            content="",
            metadata=page.metadata
        )
    
    # Use the existing normalize_text from chunker
    cleaned_text = normalize_text(page.markdown)
    
    # Remove simple markdown image links to reduce noise
    # e.g., [![alt](img.jpg)](url) -> we want to keep text, but images are noise for our extraction
    # Actually, the user says "Do NOT blindly strip content based on assumptions." 
    # and "The cleaner must prioritize preserving factual content."
    # Let's keep it very safe. We can strip empty lines and redundant whitespace.
    
    # Remove markdown images: ![alt](url)
    cleaned_text = re.sub(r'!\[.*?\]\(.*?\)', '', cleaned_text)
    
    # Ensure it's stripped
    cleaned_text = cleaned_text.strip()
    
    return CleanWebPage(
        url=page.url,
        title=page.title,
        content=cleaned_text,
        metadata=page.metadata
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
