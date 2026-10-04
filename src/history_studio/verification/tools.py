"""Claim-specific descriptions over shared search/read contracts; no dispatch or execution."""
from history_studio.research.actions import ReadRequest, SearchRequest
from history_studio.research.openai_provider import strict_schema
from .submission import VerificationSubmissionInput


def investigation_tools() -> list[dict]:
    """Build detached native function definitions for future claim-driven investigation."""
    definitions = (
        ("search_web", SearchRequest,
         "Search for independent evidence relevant to the target atomic claim only, including "
         "contradiction or qualification; do not research the whole topic. Results are leads, "
         "not accepted VerificationEvidence and not proof of source independence."),
        ("read_source", ReadRequest,
         "Read a selected source to investigate the target atomic claim only. Evaluate support, "
         "contradiction, or qualification. Returned text is candidate investigation material, "
         "not automatically accepted VerificationEvidence. The Agent will select evidence; "
         "Python will validate/extract canonical provenance from selected read spans."),
        ("submit_verification", VerificationSubmissionInput,
         "Terminally submit your semantic verification judgment for the target atomic claim only. "
         "Select supporting/contradicting source_id/span_id from canonical FactChecker reads; assess independence, "
         "uncertainty and rationale. These are evidence proposals, not accepted canonical evidence. "
         "Do not supply target identity, locator prose, versions or excerpt text. Runtime binds the claim and stops; separate finalization authenticates and extracts evidence. This is not an external tool."),
    )
    return [{"type": "function", "name": name, "description": description, "strict": True,
             "parameters": strict_schema(contract.model_json_schema())}
            for name, contract, description in definitions]
