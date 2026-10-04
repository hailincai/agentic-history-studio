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
         "Python will later validate/extract canonical provenance."),
        ("submit_verification", VerificationSubmissionInput,
         "Terminally submit your semantic verification judgment for the target atomic claim only. "
         "Select supporting/contradicting known sources and optional locators; assess independence, "
         "uncertainty and rationale. These are evidence proposals, not accepted canonical evidence. "
         "Do not supply target identity or excerpt text. Runtime binds the claim and stops; this is not an external tool."),
    )
    return [{"type": "function", "name": name, "description": description, "strict": True,
             "parameters": strict_schema(contract.model_json_schema())}
            for name, contract, description in definitions]
