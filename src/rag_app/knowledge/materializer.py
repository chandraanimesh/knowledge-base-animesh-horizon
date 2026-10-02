import logging
import uuid
import unicodedata
import re
from enum import Enum
from typing import Optional, Union
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import func, select

from rag_app.db.models import (
    KnowledgeItem,
    KnowledgeEntity,
    KnowledgeRelationship,
    KnowledgeEvidence,
)

logger = logging.getLogger(__name__)


class RelationshipProvenance(str, Enum):
    EXTRACTED = "EXTRACTED"
    INFERRED = "INFERRED"


class MaterializationCandidate(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    kind: str = "ENTITY"
    
    # Entity fields
    entity_type: Optional[str] = None
    entity_name: Optional[str] = None
    attributes: Optional[dict] = None

    # Relationship fields
    subject_name: Optional[str] = None
    subject_type: Optional[str] = None
    predicate: Optional[str] = None
    object_name: Optional[str] = None
    object_type: Optional[str] = None

    department_id: Optional[uuid.UUID] = None
    confidence: Optional[float] = None
    
    document_chunk_id: Optional[uuid.UUID] = None
    research_evidence_id: Optional[uuid.UUID] = None
    extraction_run_id: Optional[uuid.UUID] = None
    extraction_method: Optional[str] = None


class RelationshipCandidate(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    kind: str = "RELATIONSHIP"
    subject_name: str
    subject_type: str
    predicate: str
    object_name: str
    object_type: str
    provenance: RelationshipProvenance = RelationshipProvenance.EXTRACTED
    
    department_id: Optional[uuid.UUID] = None
    confidence: Optional[float] = None
    
    document_chunk_id: Optional[uuid.UUID] = None
    research_evidence_id: Optional[uuid.UUID] = None
    extraction_run_id: Optional[uuid.UUID] = None
    extraction_method: Optional[str] = None


def normalize_string(text: str) -> str:
    """
    Deterministic normalization rules for fields used in canonical identity.
    - NFKD Unicode normalization
    - Lowercase
    - Strip leading/trailing whitespace
    - Collapse repeated whitespace into a single space
    """
    if not text:
        return ""
    normalized = unicodedata.normalize('NFKD', text.strip())
    normalized = normalized.lower()
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def generate_canonical_key(kind: str, **kwargs) -> str:
    """
    Generates a deterministic canonical identity.
    - ENTITY: ENTITY:<normalized_type>:<normalized_name>
    - RELATIONSHIP: RELATIONSHIP:<normalized_subject_type>:<normalized_subject_name>:<normalized_predicate>:<normalized_object_type>:<normalized_object_name>
    """
    kind = kind.upper()
    if kind == "ENTITY":
        entity_type = normalize_string(kwargs.get("entity_type", ""))
        entity_name = normalize_string(kwargs.get("entity_name", ""))
        return f"ENTITY:{entity_type}:{entity_name}"
    
    if kind == "RELATIONSHIP":
        s_type = normalize_string(kwargs.get("subject_type", ""))
        s_name = normalize_string(kwargs.get("subject_name", ""))
        pred = normalize_string(kwargs.get("predicate", ""))
        o_type = normalize_string(kwargs.get("object_type", ""))
        o_name = normalize_string(kwargs.get("object_name", ""))
        return f"RELATIONSHIP:{s_type}:{s_name}:{pred}:{o_type}:{o_name}"
        
    raise NotImplementedError(f"Canonical key generation for {kind} is not yet implemented.")


async def materialize_knowledge(
    session: AsyncSession,
    candidate: Union[MaterializationCandidate, RelationshipCandidate],
) -> KnowledgeItem:
    """
    Materializes a single candidate (ENTITY or RELATIONSHIP) into the canonical Knowledge layer.
    Safely handles concurrent extractions using PostgreSQL ON CONFLICT.
    """
    kind = candidate.kind.upper()
    if kind == "ENTITY":
        canonical_key = generate_canonical_key(
            "ENTITY", 
            entity_type=candidate.entity_type or "", 
            entity_name=candidate.entity_name or "",
        )
        
        # 1. Upsert KnowledgeItem
        ki_stmt = insert(KnowledgeItem).values(
            canonical_key=canonical_key,
            kind="ENTITY",
            status="ACTIVE",
            department_id=candidate.department_id,
            confidence=candidate.confidence,
        ).on_conflict_do_update(
            index_elements=['canonical_key', 'department_id'],
            set_={
                'updated_at': func.now(),
                'confidence': insert(KnowledgeItem).excluded.confidence,
            }
        ).returning(KnowledgeItem)
        
        result = await session.execute(ki_stmt)
        knowledge_item = result.scalar_one()
        
        # 2. Upsert KnowledgeEntity (1:1 constraint on knowledge_item_id)
        ent_stmt = insert(KnowledgeEntity).values(
            knowledge_item_id=knowledge_item.id,
            entity_type=candidate.entity_type,
            canonical_name=candidate.entity_name,
            attributes=candidate.attributes or {},
        ).on_conflict_do_update(
            index_elements=['knowledge_item_id'],
            set_={
                'entity_type': insert(KnowledgeEntity).excluded.entity_type,
                'canonical_name': insert(KnowledgeEntity).excluded.canonical_name,
                'attributes': KnowledgeEntity.attributes.op('||')(insert(KnowledgeEntity).excluded.attributes),
            }
        )
        await session.execute(ent_stmt)
        
        # 3. Create Evidence link
        if candidate.document_chunk_id or candidate.research_evidence_id:
            ev_stmt = insert(KnowledgeEvidence).values(
                knowledge_item_id=knowledge_item.id,
                document_chunk_id=candidate.document_chunk_id,
                research_evidence_id=candidate.research_evidence_id,
                extraction_run_id=candidate.extraction_run_id,
                extraction_method=candidate.extraction_method,
                confidence=candidate.confidence,
            ).on_conflict_do_update(
                constraint='uq_knowledge_evidence_source',
                set_={
                    'extraction_run_id': insert(KnowledgeEvidence).excluded.extraction_run_id,
                    'extraction_method': insert(KnowledgeEvidence).excluded.extraction_method,
                    'confidence': insert(KnowledgeEvidence).excluded.confidence,
                }
            )
            await session.execute(ev_stmt)
            
        return knowledge_item

    elif kind == "RELATIONSHIP":
        # 1. Resolve / Create Subject KnowledgeEntity (stub if not existing)
        subj_candidate = MaterializationCandidate(
            kind="ENTITY",
            entity_type=candidate.subject_type,
            entity_name=candidate.subject_name,
            attributes={},
            department_id=candidate.department_id,
        )
        subj_item = await materialize_knowledge(session, subj_candidate)
        subj_entity_stmt = select(KnowledgeEntity).where(KnowledgeEntity.knowledge_item_id == subj_item.id)
        subj_entity = (await session.execute(subj_entity_stmt)).scalar_one()

        # 2. Resolve / Create Object KnowledgeEntity (stub if not existing)
        obj_candidate = MaterializationCandidate(
            kind="ENTITY",
            entity_type=candidate.object_type,
            entity_name=candidate.object_name,
            attributes={},
            department_id=candidate.department_id,
        )
        obj_item = await materialize_knowledge(session, obj_candidate)
        obj_entity_stmt = select(KnowledgeEntity).where(KnowledgeEntity.knowledge_item_id == obj_item.id)
        obj_entity = (await session.execute(obj_entity_stmt)).scalar_one()

        # 3. Canonical relationship key
        canonical_key = generate_canonical_key(
            "RELATIONSHIP",
            subject_type=candidate.subject_type,
            subject_name=candidate.subject_name,
            predicate=candidate.predicate,
            object_type=candidate.object_type,
            object_name=candidate.object_name,
        )

        # 4. Upsert KnowledgeItem(kind=RELATIONSHIP)
        ki_stmt = insert(KnowledgeItem).values(
            canonical_key=canonical_key,
            kind="RELATIONSHIP",
            status="ACTIVE",
            department_id=candidate.department_id,
            confidence=candidate.confidence,
        ).on_conflict_do_update(
            index_elements=['canonical_key', 'department_id'],
            set_={
                'updated_at': func.now(),
                'confidence': insert(KnowledgeItem).excluded.confidence,
            }
        ).returning(KnowledgeItem)

        result = await session.execute(ki_stmt)
        rel_item = result.scalar_one()

        # 5. Upsert KnowledgeRelationship
        rel_stmt = insert(KnowledgeRelationship).values(
            knowledge_item_id=rel_item.id,
            subject_entity_id=subj_entity.id,
            predicate=normalize_string(candidate.predicate),
            object_entity_id=obj_entity.id,
        ).on_conflict_do_update(
            index_elements=['knowledge_item_id'],
            set_={
                'subject_entity_id': insert(KnowledgeRelationship).excluded.subject_entity_id,
                'predicate': insert(KnowledgeRelationship).excluded.predicate,
                'object_entity_id': insert(KnowledgeRelationship).excluded.object_entity_id,
            }
        )
        await session.execute(rel_stmt)

        # 6. Upsert KnowledgeEvidence
        if candidate.document_chunk_id or candidate.research_evidence_id:
            ev_stmt = insert(KnowledgeEvidence).values(
                knowledge_item_id=rel_item.id,
                document_chunk_id=candidate.document_chunk_id,
                research_evidence_id=candidate.research_evidence_id,
                extraction_run_id=candidate.extraction_run_id,
                extraction_method=candidate.extraction_method,
                confidence=candidate.confidence,
            ).on_conflict_do_update(
                constraint='uq_knowledge_evidence_source',
                set_={
                    'extraction_run_id': insert(KnowledgeEvidence).excluded.extraction_run_id,
                    'extraction_method': insert(KnowledgeEvidence).excluded.extraction_method,
                    'confidence': insert(KnowledgeEvidence).excluded.confidence,
                }
            )
            await session.execute(ev_stmt)

        return rel_item

    else:
        raise NotImplementedError(f"Materialization for {candidate.kind} is not yet supported.")
