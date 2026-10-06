import logging
import re
import time
from typing import Any

from rag_app.generation.llm_factory import get_llm
from rag_app.core.observability import record_llm_call
from rag_app.schemas.rag import (
    AnswerGroundingResult,
    ConflictNotice,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)

# Currency / number regex: matches values like $8.42M, $500K, 500,000, 15%, 50ms, etc.
NUMERIC_PATTERN = re.compile(
    r"(?:\$\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:[KkMmBb]|million|billion|thousand)?|\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|[KkMmBb]|ms|s|million|billion|thousand)?\b)",
    re.IGNORECASE,
)

# Date pattern: matches October 15, Nov 20, 2026, Q3, etc.
DATE_PATTERN = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?|\b\d{4}-\d{2}-\d{2}\b|\bQ[1-4](?:\s+\d{4})?\b",
    re.IGNORECASE,
)


def _normalize_num_str(s: str) -> str:
    """Normalize numeric strings for robust matching ($500K -> 500k, 500,000 -> 500000)."""
    return s.lower().replace("$", "").replace(",", "").replace(" ", "").strip()


def _extract_numbers(text: str) -> list[str]:
    """Extract all distinct numeric and currency tokens from text."""
    raw = NUMERIC_PATTERN.findall(text)
    cleaned = []
    for t in raw:
        t_clean = t.strip(".,!?;:()[]\"' ")
        if t_clean and any(c.isdigit() for c in t_clean):
            cleaned.append(t_clean)
    return cleaned


def _is_number_in_evidence(num_val: str | int | float, evidence_text: str) -> bool:
    """Checks if a numeric value (raw, currency, thousand/million scale) exists in evidence."""
    s = str(num_val).strip()
    ev_lower = evidence_text.lower()
    norm_ev = _normalize_num_str(evidence_text)
    norm_s = _normalize_num_str(s)

    # 1. Direct string or normalized string match
    if s.lower() in ev_lower or (len(norm_s) >= 2 and norm_s in norm_ev):
        return True

    # 2. Check numerical scale equivalence
    try:
        clean_num_str = re.sub(r"[^\d.]", "", s)
        if clean_num_str:
            val_float = float(clean_num_str)
            int_val = int(val_float) if val_float.is_integer() else None

            # Direct integer match in normalized evidence (e.g. 500000.0 -> 500000)
            if int_val is not None and str(int_val) in norm_ev:
                return True

            # Sub-1000 represented in thousands (e.g. 920 -> 920k / $920K / 920,000)
            if int_val is not None and 1 <= int_val < 1000:
                if f"{int_val}k" in norm_ev or f"{int_val}thousand" in norm_ev or f"{int_val}000" in norm_ev or re.search(rf"\b{int_val}k\b|\${int_val}k\b", ev_lower):
                    return True

            # Sub-1000 represented in millions (e.g. 8.42 -> 8.42M / $8.42M)
            if f"{clean_num_str}m" in norm_ev or f"{clean_num_str}million" in norm_ev:
                return True

            # Standard thousands (e.g. 920000 -> 920k / $920K / 0.92M)
            if val_float >= 1000 and val_float % 1000 == 0:
                k_val = int(val_float / 1000)
                if f"{k_val}k" in norm_ev or f"{k_val}thousand" in norm_ev or re.search(rf"\b{k_val}k\b|\${k_val}k\b", ev_lower):
                    return True
            if val_float >= 1000000:
                m_val = f"{val_float / 1000000:g}m"
                if m_val in norm_ev:
                    return True

            # Check raw digits presence
            digits = re.sub(r"\D", "", norm_s)
            if len(digits) >= 2 and digits in re.sub(r"\D", "", evidence_text):
                return True
    except Exception:
        pass

    return False


def _extract_dates(text: str) -> list[str]:
    """Extract all distinct date patterns from text."""
    raw = DATE_PATTERN.findall(text)
    return [d.strip(".,!?;:()[]\"' ") for d in raw if d.strip(".,!?;:()[]\"' ")]


def _extract_visualization_points(viz: dict) -> list[dict[str, Any]]:
    """Extracts all data points and values from any valid VisualizationSpec schema."""
    points = []
    viz_type = viz.get("type")
    
    # 1. Bar, Line, Area (series -> data)
    if "series" in viz and isinstance(viz["series"], list):
        for s in viz["series"]:
            if isinstance(s, dict) and "data" in s and isinstance(s["data"], list):
                for item in s["data"]:
                    if isinstance(item, dict):
                        points.append(item)
    # 2. Pie, or simple direct data
    elif "data" in viz and isinstance(viz["data"], list):
        for item in viz["data"]:
            if isinstance(item, dict):
                points.append(item)
    # 3. Table (rows)
    elif "rows" in viz and isinstance(viz["rows"], list):
        for row in viz["rows"]:
            if isinstance(row, dict):
                points.append(row)
    # 4. Network (nodes + edges) - only factual labels, not internal IDs
    elif viz_type == "network":
        for node in viz.get("nodes", []):
            if isinstance(node, dict) and "label" in node:
                points.append({"label": node["label"]})
        for edge in viz.get("edges", []):
            if isinstance(edge, dict) and "label" in edge and edge["label"]:
                points.append({"label": edge["label"]})
    # 5. Timeline (events) - only date and event label
    elif viz_type == "timeline":
        for ev in viz.get("events", []):
            if isinstance(ev, dict):
                pt = {}
                if "date" in ev:
                    pt["date"] = ev["date"]
                if "label" in ev:
                    pt["label"] = ev["label"]
                points.append(pt)

    return points


def _parse_query_filter(query: str) -> tuple[str, float] | None:
    """Extracts filter operator and threshold value from query (e.g. 'above $500K' -> ('>', 500000.0))."""
    q_lower = query.lower()
    
    # Greater than / above / >
    gt_match = re.search(r"(?:above|greater than|more than|>|over)\s*\$?(\d+(?:\.\d+)?)\s*(k|m|million|billion|thousand)?", q_lower)
    if gt_match:
        val = float(gt_match.group(1))
        unit = gt_match.group(2)
        multiplier = 1000.0 if unit in ("k", "thousand") else (1000000.0 if unit in ("m", "million") else (1000000000.0 if unit in ("billion", "b") else 1.0))
        return (">", val * multiplier)

    # Less than / below / <
    lt_match = re.search(r"(?:below|less than|under|<)\s*\$?(\d+(?:\.\d+)?)\s*(k|m|million|billion|thousand)?", q_lower)
    if lt_match:
        val = float(lt_match.group(1))
        unit = lt_match.group(2)
        multiplier = 1000.0 if unit in ("k", "thousand") else (1000000.0 if unit in ("m", "million") else (1000000000.0 if unit in ("billion", "b") else 1.0))
        return ("<", val * multiplier)

    return None


class AnswerGroundingEvaluator:
    """
    Evidence-grounded answer validator and enforcement engine.
    Ensures that generated answers and visualization specifications are fully supported
    by the authorized evidence context that passed retrieval and sufficiency.
    """

    def __init__(self, llm: Any = None):
        self._llm = llm

    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def evaluate_deterministic(
        self,
        query: str,
        answer: str,
        chunks: list[RetrievedChunk],
        visualization: dict | None = None,
        conflicts: list[ConflictNotice] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        workspace_thread_context: list[dict] | None = None,
    ) -> AnswerGroundingResult:
        """
        Performs rigorous deterministic validation for numbers, dates, entities,
        visualization data points, filter constraints, and conflicting facts.
        """
        doc_text = "\n\n".join(c.text for c in chunks)
        conv_text = "\n\n".join(m.get("content", "") for m in (conversation_history or []))
        thread_text = "\n\n".join(t.get("content", "") for t in (workspace_thread_context or []))

        all_evidence = f"{doc_text}\n\n{conv_text}\n\n{thread_text}".strip()
        all_evidence_lower = all_evidence.lower()
        doc_text_lower = doc_text.lower()
        q_lower = query.lower()

        unsupported: list[str] = []
        contradictions_found: list[str] = []
        evidence_refs: list[str] = [c.chunk_id for c in chunks if c.chunk_id]

        # ----------------------------------------------------
        # 1. Check for Contradictions & Conflict Handling
        # ----------------------------------------------------
        if conflicts:
            for conf in conflicts:
                name = getattr(conf, "subject", "") or getattr(conf, "entity_name", "")
                desc = getattr(conf, "details", "") or getattr(conf, "description", "") or f"Conflict on ({getattr(conf, 'subject', '')}, {getattr(conf, 'predicate', '')})"
                contradictions_found.append(f"Conflict on entity '{name}': {desc}")

        if "different values" in q_lower or "conflicting" in q_lower:
            # If query asks about conflicting values and answer asserts certainty without noting conflict
            if "conflict" not in answer.lower() and "different" not in answer.lower() and "discrepancy" not in answer.lower():
                contradictions_found.append("Answer resolves conflicting document values without noting documented discrepancies.")

        # ----------------------------------------------------
        # 2. Deterministic Number / Metric Validation
        # ----------------------------------------------------
        answer_numbers = _extract_numbers(answer)
        for num in answer_numbers:
            norm_num = _normalize_num_str(num)
            # Only ignore single-digit enumeration formatting when matching list formatting e.g. "1." or "(1)"
            if re.search(rf"(?:^|\n)\s*(?:\({re.escape(norm_num)}\)|{re.escape(norm_num)}\.)\s+", answer):
                continue

            if not _is_number_in_evidence(num, all_evidence):
                unsupported.append(f"Numeric value '{num}' in answer is not supported by retrieved evidence.")

        # ----------------------------------------------------
        # 3. Deterministic Date / Deadline Validation
        # ----------------------------------------------------
        answer_dates = _extract_dates(answer)
        for date_str in answer_dates:
            d_norm = date_str.lower()
            if d_norm not in all_evidence_lower:
                tokens = [t.strip("stndrh,") for t in d_norm.split()]
                all_tokens_match = True
                for t in tokens:
                    if len(t) >= 1:
                        # Match whole word to avoid '20' matching inside '2026'
                        if not re.search(rf"\b{re.escape(t)}(?:st|nd|rd|th)?\b", all_evidence_lower):
                            all_tokens_match = False
                            break
                if not all_tokens_match:
                    unsupported.append(f"Date claim '{date_str}' in answer is not supported by retrieved evidence.")

        # ----------------------------------------------------
        # 4. Entity / Terminology Validation
        # ----------------------------------------------------
        answer_words = re.findall(r"\b[A-Z][a-zA-Z0-9_-]+\b", answer)
        common_words = {
            "The", "This", "That", "These", "Those", "Here", "What", "When", "Where", "Who", "Why", "How",
            "There", "Worker", "Section", "Table", "Deal", "Failed", "Mandatory", "Audit", "Database", "Version",
            "Node", "API", "Q1", "Q2", "Q3", "Q4", "SOC2", "ACID", "REST", "JSON", "SQL", "LRU", "RAM", "EMEA"
        }
        for word in answer_words:
            if word not in common_words and word.lower() not in all_evidence_lower and word.lower() not in q_lower:
                unsupported.append(f"Entity/term '{word}' in answer is not supported by retrieved evidence.")

        # ----------------------------------------------------
        # 5. Visualization Data Point & Filter Validation
        # ----------------------------------------------------
        if visualization and isinstance(visualization, dict):
            points = _extract_visualization_points(visualization)
            for idx, pt in enumerate(points):
                for k, v in pt.items():
                    if v is None:
                        continue
                    v_str = str(v).strip()
                    if not v_str or v_str.lower() in {"null", "none"}:
                        continue

                    # Validate categorical labels / names (length > 2)
                    if isinstance(v, str) and len(v_str) > 2 and not v_str.replace(".", "").isdigit():
                        v_norm = v_str.lower()
                        if v_norm not in all_evidence_lower and v_norm not in q_lower:
                            tokens = [w for w in re.findall(r"\w+", v_norm) if len(w) > 2]
                            significant = [w for w in tokens if w not in {"the", "and", "for", "with", "deal", "deals", "item", "row", "discussion", "details", "info", "overview"}]
                            check_tokens = significant if significant else tokens
                            if not all(t in all_evidence_lower or t in q_lower for t in check_tokens):
                                unsupported.append(f"Visualization item {idx + 1} contains ungrounded entity/label '{v_str}'.")

                    # Validate numerical measures
                    if isinstance(v, (int, float)) or (isinstance(v, str) and any(c.isdigit() for c in v_str)):
                        if not _is_number_in_evidence(v, all_evidence):
                            unsupported.append(f"Visualization item {idx + 1} measure '{v}' is not supported by evidence.")

            # Filter violation validation: numeric (e.g. "above $500K")
            q_filter = _parse_query_filter(query)
            if q_filter:
                op, threshold = q_filter
                for idx, pt in enumerate(points):
                    for k, v in pt.items():
                        try:
                            if isinstance(v, (int, float)):
                                val_num = float(v)
                            elif isinstance(v, str):
                                digits_only = re.sub(r"[^\d.]", "", v)
                                if digits_only:
                                    val_num = float(digits_only)
                                else:
                                    continue
                            else:
                                continue

                            # Adjust for thousands representation
                            if val_num < 1000 and threshold >= 1000:
                                val_num *= 1000.0

                            if op == ">" and val_num < threshold:
                                unsupported.append(
                                    f"Visualization item {idx + 1} violates query filter: value {v} is below threshold {threshold}."
                                )
                            elif op == "<" and val_num > threshold:
                                unsupported.append(
                                    f"Visualization item {idx + 1} violates query filter: value {v} is above threshold {threshold}."
                                )
                        except Exception:
                            pass

            # Filter violation validation: categorical (e.g. "in Ahmedabad")
            loc_match = re.search(r"\bin\s+([A-Z][a-zA-Z]+)\b", query)
            if loc_match:
                target_loc = loc_match.group(1).lower()
                for idx, pt in enumerate(points):
                    for k, v in pt.items():
                        if isinstance(v, str) and k.lower() in ("location", "city", "region", "country"):
                            if v.lower() != target_loc:
                                unsupported.append(
                                    f"Visualization item {idx + 1} violates location filter '{loc_match.group(1)}' with '{v}'."
                                )

        is_grounded = len(unsupported) == 0 and len(contradictions_found) == 0

        reason_str = (
            "All claims, dates, numbers, and visualization data points are grounded in authorized evidence."
            if is_grounded
            else f"Detected {len(unsupported)} unsupported claim(s) and {len(contradictions_found)} contradiction(s)."
        )

        return AnswerGroundingResult(
            is_grounded=is_grounded,
            confidence=0.95 if is_grounded else 0.4,
            unsupported_claims=unsupported,
            evidence_refs=evidence_refs,
            contradictions=contradictions_found,
            reason=reason_str,
        )

    def repair_visualization(
        self,
        visualization: dict,
        query: str,
        chunks: list[RetrievedChunk],
        conversation_history: list[dict[str, str]] | None = None,
        workspace_thread_context: list[dict] | None = None,
    ) -> dict | None:
        """
        Deterministically repairs a visualization by filtering out ungrounded data points
        or filter-violating rows. If the remaining data is insufficient for the chart type, returns None.
        """
        doc_text = "\n\n".join(c.text for c in chunks)
        conv_text = "\n\n".join(m.get("content", "") for m in (conversation_history or []))
        thread_text = "\n\n".join(t.get("content", "") for t in (workspace_thread_context or []))
        all_evidence = f"{doc_text}\n\n{conv_text}\n\n{thread_text}".strip().lower()
        q_lower = query.lower()

        q_filter = _parse_query_filter(query)
        import copy
        repaired = copy.deepcopy(visualization)
        viz_type = repaired.get("type")

        def is_point_valid(pt: dict) -> bool:
            # Check label
            for k, v in pt.items():
                if isinstance(v, str) and len(v.strip()) > 2 and not v.strip().replace(".", "").isdigit():
                    v_norm = v.strip().lower()
                    if v_norm not in all_evidence and v_norm not in q_lower:
                        tokens = [w for w in re.findall(r"\w+", v_norm) if len(w) > 2]
                        significant = [w for w in tokens if w not in {"the", "and", "for", "with", "deal", "deals", "item", "row", "discussion", "details", "info", "overview"}]
                        check_tokens = significant if significant else tokens
                        if not all(t in all_evidence or t in q_lower for t in check_tokens):
                            return False
                if isinstance(v, (int, float)) or (isinstance(v, str) and any(c.isdigit() for c in str(v))):
                    if not _is_number_in_evidence(v, all_evidence):
                        return False
            # Check filter
            if q_filter:
                op, threshold = q_filter
                for k, v in pt.items():
                    try:
                        if isinstance(v, (int, float)):
                            val_num = float(v)
                        elif isinstance(v, str):
                            digits_only = re.sub(r"[^\d.]", "", v)
                            if digits_only:
                                val_num = float(digits_only)
                            else:
                                continue
                        else:
                            continue
                        if val_num < 1000 and threshold >= 1000:
                            val_num *= 1000.0
                        if op == ">" and val_num < threshold:
                            return False
                        elif op == "<" and val_num > threshold:
                            return False
                    except Exception:
                        pass
            return True

        if "series" in repaired and isinstance(repaired["series"], list):
            for s in repaired["series"]:
                if isinstance(s, dict) and "data" in s and isinstance(s["data"], list):
                    s["data"] = [item for item in s["data"] if isinstance(item, dict) and is_point_valid(item)]
            # If any series is now empty, chart is invalid
            if any(len(s.get("data", [])) == 0 for s in repaired["series"]):
                return None
            return repaired

        elif "data" in repaired and isinstance(repaired["data"], list):
            repaired["data"] = [item for item in repaired["data"] if isinstance(item, dict) and is_point_valid(item)]
            if viz_type == "pie" and len(repaired["data"]) < 2:
                return None
            if len(repaired["data"]) == 0:
                return None
            return repaired

        elif "rows" in repaired and isinstance(repaired["rows"], list):
            repaired["rows"] = [row for row in repaired["rows"] if isinstance(row, dict) and is_point_valid(row)]
            if len(repaired["rows"]) == 0:
                return None
            return repaired

        elif viz_type == "network":
            if "nodes" in repaired and isinstance(repaired["nodes"], list):
                valid_nodes = [
                    n for n in repaired["nodes"]
                    if isinstance(n, dict) and ("label" not in n or is_point_valid({"label": n["label"]}))
                ]
                valid_node_ids = {n.get("id") for n in valid_nodes if n.get("id")}
                repaired["nodes"] = valid_nodes
                if "edges" in repaired and isinstance(repaired["edges"], list):
                    repaired["edges"] = [
                        e for e in repaired["edges"]
                        if isinstance(e, dict) and e.get("source") in valid_node_ids and e.get("target") in valid_node_ids
                    ]
                if len(repaired["nodes"]) == 0:
                    return None
                return repaired

        elif viz_type == "timeline":
            if "events" in repaired and isinstance(repaired["events"], list):
                repaired["events"] = [
                    ev for ev in repaired["events"]
                    if isinstance(ev, dict) and (ev.get("date", "").lower() in all_evidence or is_point_valid({"date": ev.get("date", "")}))
                ]
                if len(repaired["events"]) == 0:
                    return None
                return repaired

        return None

    async def evaluate(
        self,
        query: str,
        answer: str,
        chunks: list[RetrievedChunk],
        visualization: dict | None = None,
        conflicts: list[ConflictNotice] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        workspace_thread_context: list[dict] | None = None,
    ) -> AnswerGroundingResult:
        """
        Asynchronously validates answer grounding combining deterministic rules
        and structured LLM semantic claim verification.
        """
        # 1. Deterministic validation first (catches numbers, dates, filter violations, row fabrications)
        det_result = self.evaluate_deterministic(
            query=query,
            answer=answer,
            chunks=chunks,
            visualization=visualization,
            conflicts=conflicts,
            conversation_history=conversation_history,
            workspace_thread_context=workspace_thread_context,
        )

        if not det_result.is_grounded:
            return det_result

        # 2. If deterministic checks pass, run structured LLM claim verification on nuanced text
        try:
            llm = self._get_llm()
            doc_context = "\n\n".join(f"[Chunk {c.chunk_id}]:\n{c.text}" for c in chunks)
            conv_context = "\n\n".join(f"[{m.get('role')}]: {m.get('content')}" for m in (conversation_history or []))
            
            full_context = f"DOCUMENT EVIDENCE:\n{doc_context}\n\nCONVERSATION HISTORY:\n{conv_context}".strip()

            system_prompt = (
                "You are an expert answer grounding evaluator.\n"
                "Verify whether all factual statements, requirements, comparisons, and claims in the generated answer "
                "are directly supported by or logically entailed by the provided evidence.\n\n"
                "Rules:\n"
                "1. Every factual assertion (entities, requirements, comparisons, conclusions) must be grounded in the evidence.\n"
                "2. If the answer invents requirements, responsibilities, or facts not mentioned in the evidence, set is_grounded=False and list the unsupported claims.\n"
                "3. If the answer accurately reflects the evidence or appropriately states document limitations, set is_grounded=True.\n"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"EVIDENCE CONTEXT:\n{full_context}\n\nUSER QUESTION:\n{query}\n\nGENERATED ANSWER:\n{answer}"},
            ]

            structured_llm = llm.llm.with_structured_output(
                AnswerGroundingResult,
                include_raw=True,
            )

            _eval_start = time.perf_counter()
            inv_res = await structured_llm.ainvoke(messages)
            _eval_latency = (time.perf_counter() - _eval_start) * 1000

            raw_msg = inv_res.get("raw") if isinstance(inv_res, dict) else None
            usage_meta = getattr(raw_msg, "usage_metadata", None) or {}
            record_llm_call(
                stage_name="generation",
                usage=usage_meta,
                latency_ms=_eval_latency,
                model_name=getattr(llm.llm, "model_name", None),
            )

            result: AnswerGroundingResult = inv_res.get("parsed") if isinstance(inv_res, dict) else inv_res
            if result is not None:
                # Merge any deterministic findings
                if det_result.unsupported_claims:
                    result.unsupported_claims = list(set(result.unsupported_claims + det_result.unsupported_claims))
                    result.is_grounded = False
                if not result.evidence_refs:
                    result.evidence_refs = det_result.evidence_refs
                return result

            return det_result

        except Exception as exc:
            logger.warning(
                "answer_grounding_llm_failed_using_deterministic",
                extra={
                    "event": "answer_grounding_llm_failed",
                    "error": str(exc),
                    "query": query,
                },
            )
            return det_result
