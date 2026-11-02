"""Mock legal research tools over the fictional Northbridge corpus."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

CORPUS_PATH = Path(__file__).with_name("corpus.json")


def load_corpus() -> dict[str, Any]:
    with CORPUS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


class LegalTools:
    """Closed-corpus tools. Optional force_error simulates a broken lookup."""

    def __init__(self, force_error_ids: set[str] | None = None) -> None:
        self.corpus = load_corpus()
        self.force_error_ids = force_error_ids or set()
        self.call_log: list[dict[str, Any]] = []

    def _log(self, name: str, args: dict[str, Any], result: Any) -> Any:
        self.call_log.append({"tool": name, "args": args, "result": result})
        return result

    def search_corpus(self, query: str) -> dict[str, Any]:
        q = query.lower()
        hits = []
        for s in self.corpus["statutes"]:
            blob = " ".join(
                [
                    s["id"],
                    s["title"],
                    s["text"],
                    " ".join(s.get("keywords", [])),
                ]
            ).lower()
            score = sum(1 for token in re.findall(r"[a-z0-9-]+", q) if token in blob)
            if score:
                hits.append(
                    {
                        "id": s["id"],
                        "title": s["title"],
                        "status": s["status"],
                        "score": score,
                        "snippet": s["text"][:160] + ("..." if len(s["text"]) > 160 else ""),
                    }
                )
        hits.sort(key=lambda h: h["score"], reverse=True)
        return self._log("search_corpus", {"query": query}, {"hits": hits[:5]})

    def read_section(self, section_id: str) -> dict[str, Any]:
        sid = section_id.strip().upper()
        if sid in self.force_error_ids or sid == "NB-XX-99":
            return self._log(
                "read_section",
                {"section_id": section_id},
                {
                    "error": "TOOL_ERROR",
                    "message": f"No document found for {sid}. Retrieval backend returned empty.",
                },
            )
        for s in self.corpus["statutes"]:
            if s["id"].upper() == sid:
                return self._log("read_section", {"section_id": section_id}, {"section": s})
        return self._log(
            "read_section",
            {"section_id": section_id},
            {
                "error": "NOT_FOUND",
                "message": f"Section {sid} is not in the Northbridge demo corpus.",
            },
        )

    def verify_citation(self, section_id: str, claim: str) -> dict[str, Any]:
        doc = self.read_section(section_id)
        if "error" in doc:
            result = {
                "verified": False,
                "reason": doc.get("message", "source unavailable"),
                "section_id": section_id,
                "claim": claim,
            }
            return self._log("verify_citation", {"section_id": section_id, "claim": claim}, result)

        text = doc["section"]["text"].lower()
        status = doc["section"]["status"]
        claim_l = claim.lower()
        numbers_in_claim = set(re.findall(r"\d+", claim_l))
        numbers_in_text = set(re.findall(r"\d+", text))

        if status == "repealed":
            result = {
                "verified": False,
                "reason": f"{section_id} is repealed"
                + (
                    f" as of {doc['section'].get('repealed_on')}"
                    if doc["section"].get("repealed_on")
                    else ""
                ),
                "section_id": section_id,
                "claim": claim,
                "section_status": status,
                "section_text": doc["section"]["text"],
            }
            return self._log("verify_citation", {"section_id": section_id, "claim": claim}, result)

        keywords = [k.lower() for k in doc["section"].get("keywords", [])]
        keyword_hits = [k for k in keywords if k in claim_l]
        number_conflict = bool(numbers_in_claim) and not numbers_in_claim.issubset(numbers_in_text)

        if number_conflict:
            result = {
                "verified": False,
                "reason": "Claimed number(s) do not appear in the statute text.",
                "section_id": section_id,
                "claim": claim,
                "section_status": status,
                "section_text": doc["section"]["text"],
                "keyword_hits": keyword_hits,
            }
        elif keyword_hits or any(tok in text for tok in claim_l.split() if len(tok) > 4):
            result = {
                "verified": True,
                "reason": "Claim is consistent with retrieved statute text.",
                "section_id": section_id,
                "claim": claim,
                "section_status": status,
                "section_text": doc["section"]["text"],
                "keyword_hits": keyword_hits,
            }
        else:
            result = {
                "verified": False,
                "reason": "Claim is not supported by the retrieved statute text.",
                "section_id": section_id,
                "claim": claim,
                "section_status": status,
                "section_text": doc["section"]["text"],
                "keyword_hits": keyword_hits,
            }
        return self._log("verify_citation", {"section_id": section_id, "claim": claim}, result)

    def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "search_corpus":
            return self.search_corpus(args.get("query", ""))
        if name == "read_section":
            return self.read_section(args.get("section_id", ""))
        if name == "verify_citation":
            return self.verify_citation(args.get("section_id", ""), args.get("claim", ""))
        return {"error": "UNKNOWN_TOOL", "message": f"No tool named {name}"}
