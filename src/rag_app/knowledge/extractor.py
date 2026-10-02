import logging
import uuid
from typing import Optional, List, Union

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from rag_app.db.models import DocumentChunk, Document
from rag_app.generation.llm_factory import get_llm
from rag_app.knowledge.materializer import (
    MaterializationCandidate,
    RelationshipCandidate,
    RelationshipProvenance,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a strict internal document data extraction system.
Your job is to extract factual knowledge (entities and explicitly stated relationships) from the provided document chunk.

# Rules
1. Extract ONLY information explicitly supported by the input chunk. DO NOT hallucinate.
2. The extracted facts must be returned as:
   - Structured entities (e.g., Company, Person, Product) with their attributes (e.g., email, phone, location).
   - Structured relationships between entities (subject -> predicate -> object) ONLY when explicitly stated in the text.
3. Grounding rules for relationships:
   - Extract a relationship ONLY when the chunk explicitly asserts it.
     Example: "Acme hired Alice as CTO." -> subject_name="Alice", subject_type="PERSON", predicate="WORKS_AT", object_name="Acme", object_type="ORGANIZATION"
   - DO NOT infer unstated relationships, ownership, causality, employment beyond explicit wording, or relationships from general knowledge.
     Example: If the chunk says "Acme hired Alice as CTO", DO NOT emit predicate="OWNS".
4. DO NOT infer unstated facts, industry, country, or missing values.
5. DO NOT invent entities or relationships from semantic similarity or use outside/general knowledge.
6. Only emit attributes directly supported by the chunk.
   For example, if the chunk says: "Acme is headquartered in Ahmedabad."
   Produce entity: entity_name="Acme", entity_type="ORGANIZATION", attributes={"city": "Ahmedabad"}
   Do NOT infer country="India" or industry="Technology".
7. If no relevant factual information exists in the chunk, output empty entities and relationships lists.
"""


class InternalExtractedEntity(BaseModel):
    """A single structured entity extracted from an internal document chunk."""
    entity_name: str = Field(..., description="Name of the entity (e.g., Company Name, Person, Product).")
    entity_type: str = Field(..., description="Type of the entity (e.g., ORGANIZATION, PERSON, LOCATION).")
    attributes: dict = Field(default_factory=dict, description="Structured attributes ONLY if explicitly stated in the chunk.")


class InternalExtractedRelationship(BaseModel):
    """A single structured relationship extracted from an internal document chunk."""
    subject_name: str = Field(..., description="Name of the subject entity (e.g., Alice Smith, Acme Corp).")
    subject_type: str = Field(..., description="Type of the subject entity (e.g., PERSON, ORGANIZATION).")
    predicate: str = Field(..., description="The explicit relationship verb or phrase connecting subject to object (e.g., WORKS_AT, FOUNDED, LOCATED_IN, DEVELOPED).")
    object_name: str = Field(..., description="Name of the object entity (e.g., Acme Corp, Titan AI).")
    object_type: str = Field(..., description="Type of the object entity (e.g., ORGANIZATION, PRODUCT, LOCATION).")


class InternalExtractionResult(BaseModel):
    """Collection of entities and explicit relationships extracted from the chunk."""
    entities: list[InternalExtractedEntity] = Field(default_factory=list, description="List of extracted entities.")
    relationships: list[InternalExtractedRelationship] = Field(default_factory=list, description="List of explicitly stated relationships between entities.")


class KnowledgeExtractorService:
    def __init__(self):
        # Service is stateless, relies on provider abstraction
        self.provider = get_llm()
        self.structured_llm = self.provider.llm.with_structured_output(InternalExtractionResult)
        
    async def extract_chunk(self, chunk: DocumentChunk, document: Document) -> List[Union[MaterializationCandidate, RelationshipCandidate]]:
        """
        Extract entities and explicit relationships from a single document chunk.
        Returns a list of MaterializationCandidates and RelationshipCandidates.
        Does NOT persist directly; candidates must be passed to the materializer by the caller.
        """
        text = chunk.text.strip()
        if not text:
            return []
            
        # Ignore extremely short chunks that cannot meaningfully contain an entity and context
        if len(text) < 10:
            return []
            
        metadata = chunk.metadata_ or {}
        element_type = metadata.get("element_type", "unknown")
        
        # Build context
        context_parts = []
        if element_type:
            context_parts.append(f"Element Type: {element_type}")
        if "header_1" in metadata:
            context_parts.append(f"Section 1: {metadata['header_1']}")
        if "header_2" in metadata:
            context_parts.append(f"Section 2: {metadata['header_2']}")
            
        context_str = "\n".join(context_parts)
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"CHUNK METADATA:\n{context_str}\n\nCHUNK CONTENT:\n{text}")
        ]
        
        try:
            result: InternalExtractionResult = await self.structured_llm.ainvoke(messages)
            
            candidates: list[Union[MaterializationCandidate, RelationshipCandidate]] = []
            
            # 1. Process extracted entities
            for entity in result.entities:
                if not (entity.entity_name and entity.entity_type):
                    continue
                candidate = MaterializationCandidate(
                    kind="ENTITY",
                    entity_type=entity.entity_type.strip(),
                    entity_name=entity.entity_name.strip(),
                    attributes=entity.attributes or {},
                    department_id=document.department_id,
                    document_chunk_id=chunk.id,
                    extraction_method="internal_llm",
                )
                candidates.append(candidate)
                
            # 2. Process extracted relationships
            for rel in result.relationships:
                if not (rel.subject_name and rel.subject_type and rel.predicate and rel.object_name and rel.object_type):
                    continue
                rel_candidate = RelationshipCandidate(
                    kind="RELATIONSHIP",
                    subject_name=rel.subject_name.strip(),
                    subject_type=rel.subject_type.strip(),
                    predicate=rel.predicate.strip(),
                    object_name=rel.object_name.strip(),
                    object_type=rel.object_type.strip(),
                    provenance=RelationshipProvenance.EXTRACTED,
                    department_id=document.department_id,
                    document_chunk_id=chunk.id,
                    extraction_method="internal_llm",
                )
                candidates.append(rel_candidate)

            return candidates
            
        except Exception as e:
            logger.error(
                "chunk_extraction_failed",
                extra={
                    "event": "chunk_extraction_failed",
                    "chunk_id": str(chunk.id),
                    "document_id": str(document.id),
                    "error": str(e),
                },
            )
            return []
