"""Terminal narrative handoff schema, never an external research capability."""
from history_studio.research.openai_provider import strict_schema
from .submission import ScriptSubmission


def script_tools() -> list[dict]:
    return [{"type": "function", "name": "submit_script", "strict": True,
             "description": "Terminally submit the narrative proposal. Supply only Agent-owned "
             "ScriptSubmission fields. Runtime authenticates sections, beats, selected fact identities and provenance. "
             "This is a semantic handoff, not an external tool.",
             "parameters": strict_schema(ScriptSubmission.model_json_schema())}]
