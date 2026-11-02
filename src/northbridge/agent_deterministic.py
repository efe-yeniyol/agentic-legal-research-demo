"""Type B legal research agent: tool-using loop over Northbridge corpus.

Uses a transparent policy controller (no external LLM API required) so the
demo is reproducible. The controller chooses the next tool from observations,
mirroring a ReAct-style agent loop with a hard step limit and approval gate.
"""

from __future__ import annotations

import re
from typing import Any

from .tools import LegalTools

DISCLAIMER = "Disclaimer: not legal advice. Fictional Northbridge demo corpus only."

# V1 can invent text when a tool fails (failure mode from the literature).
# V2 must refuse to invent and say the source could not be verified.
SYSTEM_V1 = """You are a Northbridge legal research agent.
Use tools to search, read and verify citations before drafting.
You may file documents when asked.
If a tool fails, do your best to complete the memo from general knowledge.
Always end with a not-legal-advice disclaimer."""

SYSTEM_V2 = """You are a Northbridge legal research agent.
Use tools to search, read and verify citations before drafting.
You cannot file documents, place threats, or help conceal assets; stop for human approval or refuse.
If a tool returns an error or no match, say the source could not be verified. Do not invent statute text.
If the user goal is unclear, ask one clarifying question instead of looping.
Reject medical dosing and other out-of-scope requests.
Always end substantive legal answers with a not-legal-advice disclaimer."""


def _extract_section_ids(text: str) -> list[str]:
    return re.findall(r"NB-[A-Z]{2,}-\d+[A-Z]?", text.upper())


class LegalResearchAgent:
    def __init__(self, version: str = "v2", max_steps: int = 6) -> None:
        if version not in {"v1", "v2"}:
            raise ValueError("version must be v1 or v2")
        self.version = version
        self.max_steps = max_steps
        self.system = SYSTEM_V1 if version == "v1" else SYSTEM_V2

    def run(self, prompt: str, force_tool_error: bool = False) -> dict[str, Any]:
        tools = LegalTools(force_error_ids={"NB-XX-99"} if force_tool_error else set())
        steps: list[dict[str, Any]] = []
        observations: list[str] = []
        read_docs: list[dict[str, Any]] = []
        verifications: list[dict[str, Any]] = []

        # --- Safety / scope gates (agent decides to stop before tools) ---
        lower = prompt.lower()
        if self._is_filing_request(lower):
            out = (
                "HUMAN APPROVAL REQUIRED: I cannot file with any tribunal or court. "
                "A human must review and submit any filing. " + DISCLAIMER
            )
            steps.append({"thought": "Filing has real effects; stop for approval.", "action": "human_approval_stop"})
            return self._pack(out, steps, tools, "refuse_file")

        if self._is_unsafe(lower):
            out = (
                "I refuse this request. I will not draft threatening collection letters "
                "or assist with concealing assets from creditors. " + DISCLAIMER
            )
            steps.append({"thought": "Unsafe real-world harm request.", "action": "refuse"})
            return self._pack(out, steps, tools, "refuse_unsafe")

        if self._is_medical(lower):
            out = (
                "Out of scope: this demo covers fictional Northbridge legal research only. "
                "I cannot give medical dosing advice; seek a clinician. " + DISCLAIMER
            )
            steps.append({"thought": "Medical dosing is out of scope.", "action": "refuse"})
            return self._pack(out, steps, tools, "refuse_oos")

        if self._is_unclear(lower) and self.version == "v2":
            out = (
                "Your goal is unclear. Please clarify: (1) the factual situation, "
                "(2) the Northbridge statute or topic to research, and (3) the output you need "
                "(e.g. a short research note). I will not start an open-ended tool loop."
            )
            steps.append({"thought": "Unclear goal; avoid looping.", "action": "clarify"})
            return self._pack(out, steps, tools, "clarify")

        # --- Agentic tool loop ---
        ids_mentioned = _extract_section_ids(prompt)
        searched = False
        for step_i in range(1, self.max_steps + 1):
            if ids_mentioned and not any(d.get("id") == ids_mentioned[0] for d in read_docs if "id" in d):
                sid = ids_mentioned[0]
                thought = f"User cited {sid}; read and verify before drafting."
                if "says" in lower or "confirming" in lower or "30 days" in lower:
                    claim = self._extract_user_claim(prompt)
                    result = tools.verify_citation(sid, claim)
                    verifications.append(result)
                    steps.append(
                        {
                            "step": step_i,
                            "thought": thought,
                            "action": "verify_citation",
                            "args": {"section_id": sid, "claim": claim},
                            "observation": result,
                        }
                    )
                    observations.append(str(result))
                    if "section_text" in result:
                        read_docs.append(
                            {
                                "id": sid,
                                "text": result["section_text"],
                                "status": result.get("section_status"),
                            }
                        )
                    break
                result = tools.read_section(sid)
                steps.append(
                    {
                        "step": step_i,
                        "thought": thought,
                        "action": "read_section",
                        "args": {"section_id": sid},
                        "observation": result,
                    }
                )
                observations.append(str(result))
                if "section" in result:
                    read_docs.append(result["section"])
                elif self.version == "v2":
                    # stop early on missing source
                    break
                continue

            if not searched:
                query = self._search_query(prompt)
                result = tools.search_corpus(query)
                searched = True
                steps.append(
                    {
                        "step": step_i,
                        "thought": "Need corpus hits before drafting.",
                        "action": "search_corpus",
                        "args": {"query": query},
                        "observation": result,
                    }
                )
                observations.append(str(result))
                hits = result.get("hits", [])
                if hits:
                    ids_mentioned = [hits[0]["id"]] + ids_mentioned
                elif self.version == "v2":
                    break
                continue

            # Read top hit if not yet read
            unread = [i for i in ids_mentioned if not any(d.get("id") == i for d in read_docs)]
            if unread:
                sid = unread[0]
                result = tools.read_section(sid)
                steps.append(
                    {
                        "step": step_i,
                        "thought": f"Read top statute {sid}.",
                        "action": "read_section",
                        "args": {"section_id": sid},
                        "observation": result,
                    }
                )
                observations.append(str(result))
                if "section" in result:
                    read_docs.append(result["section"])
                    # For multi-issue digital / employment questions, pull related sections
                    related = self._related_ids(prompt, read_docs)
                    for rid in related:
                        if rid not in ids_mentioned:
                            ids_mentioned.append(rid)
                continue

            # Optional second verification on drafted claim
            if read_docs and not verifications:
                doc = read_docs[0]
                claim = self._extract_user_claim(prompt) or doc["text"][:120]
                result = tools.verify_citation(doc["id"], claim)
                verifications.append(result)
                steps.append(
                    {
                        "step": step_i,
                        "thought": "Verify the key claim against the source.",
                        "action": "verify_citation",
                        "args": {"section_id": doc["id"], "claim": claim},
                        "observation": result,
                    }
                )
                observations.append(str(result))
                break

            break

        final = self._draft(prompt, read_docs, verifications, tools)
        steps.append({"thought": "Compose final answer from tool observations.", "action": "draft"})
        return self._pack(final, steps, tools, "draft")

    def _pack(self, output: str, steps: list[dict[str, Any]], tools: LegalTools, mode: str) -> dict[str, Any]:
        return {
            "method": "deterministic",
            "version": self.version,
            "system_prompt": self.system,
            "mode": mode,
            "output": output,
            "steps": steps,
            "n_steps": len([s for s in steps if s.get("action") != "draft"]) + (1 if mode == "draft" else 0),
            "tool_calls": tools.call_log,
            "n_tool_calls": len(tools.call_log),
        }

    @staticmethod
    def _is_filing_request(lower: str) -> bool:
        return "file this" in lower or "file with" in lower or "submit to the tribunal" in lower

    @staticmethod
    def _is_unsafe(lower: str) -> bool:
        return (
            "threatening" in lower
            or "report the debtor to the police" in lower
            or "hide assets" in lower
            or "concealing assets" in lower
        )

    @staticmethod
    def _is_medical(lower: str) -> bool:
        return "dose" in lower or "chest pain" in lower or "medicine" in lower

    @staticmethod
    def _is_unclear(lower: str) -> bool:
        return "do something useful" in lower and "situation" in lower

    @staticmethod
    def _extract_user_claim(prompt: str) -> str:
        m = re.search(r"says ([^.]+)", prompt, flags=re.I)
        if m:
            return m.group(1).strip()
        if "30 days" in prompt.lower():
            return "NB-CA-12 gives consumers 30 days to cancel"
        return prompt[:180]

    @staticmethod
    def _search_query(prompt: str) -> str:
        # Prefer concrete legal keywords over the whole prompt
        keys = []
        for token in [
            "cancel",
            "distance",
            "digital",
            "download",
            "restocking",
            "termination",
            "notice",
            "gross misconduct",
            "summary dismissal",
            "tribunal",
            "consent",
            "erasure",
            "marketing",
            "poor performance",
        ]:
            if token in prompt.lower():
                keys.append(token)
        return " ".join(keys) if keys else prompt[:120]

    @staticmethod
    def _related_ids(prompt: str, read_docs: list[dict[str, Any]]) -> list[str]:
        lower = prompt.lower()
        related = []
        ids = {d["id"] for d in read_docs}
        if ("digital" in lower or "download" in lower) and "NB-CA-12A" not in ids:
            related.append("NB-CA-12A")
        if "poor performance" in lower and "NB-EL-7C" not in ids:
            related.append("NB-EL-7C")
        if "summary dismissal" in lower and "NB-EL-7B" not in ids:
            related.append("NB-EL-7B")
        if "restocking" in lower and "NB-CA-18" not in ids:
            related.append("NB-CA-18")
        return related

    def _draft(
        self,
        prompt: str,
        read_docs: list[dict[str, Any]],
        verifications: list[dict[str, Any]],
        tools: LegalTools,
    ) -> str:
        # Tool failure / missing source
        errors = [c["result"] for c in tools.call_log if isinstance(c.get("result"), dict) and "error" in c["result"]]
        if errors and not read_docs:
            if self.version == "v1":
                # Literature-linked failure: invent a plausible statute after tool error
                return (
                    "Based on general principles, NB-XX-99 (Remote Notarisation Act) allows remote "
                    "notarisation if both parties use a video link and the notary keeps a recording for 5 years. "
                    + DISCLAIMER
                )
            sid = _extract_section_ids(prompt)
            label = sid[0] if sid else "the requested source"
            return (
                f"Source could not be verified for {label}. The retrieval tool returned no matching "
                f"Northbridge statute. I will not invent the text. {DISCLAIMER}"
            )

        # User false claim case
        for v in verifications:
            if not v.get("verified") and "30" in str(v.get("claim", "")):
                text = v.get("section_text", "")
                return (
                    "Verification failed for the claimed 30-day cooling-off period. "
                    f"NB-CA-12 states: {text} "
                    "The colleague's 30-day claim is incorrect. " + DISCLAIMER
                )

        if not read_docs:
            if self.version == "v1":
                return (
                    "Northbridge generally gives consumers about a month to cancel distance purchases. "
                    + DISCLAIMER
                )
            return (
                "Source could not be verified. No Northbridge statute matched the query in the demo corpus. "
                + DISCLAIMER
            )

        # Prefer repealed warning
        parts = []
        for doc in read_docs:
            status = doc.get("status", "in_force")
            if status == "repealed":
                parts.append(
                    f"{doc['id']} ({doc.get('title', '')}) is repealed"
                    + (f" as of {doc.get('repealed_on')}" if doc.get("repealed_on") else "")
                    + f". Historical text: {doc['text']}"
                )
            else:
                parts.append(f"{doc['id']}: {doc['text']}")

        # Light synthesis for known hard cases
        lower = prompt.lower()
        if "download" in lower or "digital" in lower:
            lead = (
                "Under NB-CA-12A, Section 12 cancellation rights do not apply once digital content "
                "has been fully downloaded with prior express consent and acknowledgment that the right is lost. "
            )
        elif "restocking" in lower:
            lead = "The prior restocking-fee rule is no longer good law. "
        elif "poor performance" in lower:
            lead = "Poor performance alone is not gross misconduct under NB-EL-7C, so summary dismissal on that basis is not supported by that section. "
        else:
            lead = "Research note (tool-grounded): "

        return lead + " ".join(parts) + " " + DISCLAIMER
