"""Terminal visual handoff schema, without external capabilities."""
from history_studio.research.openai_provider import strict_schema
from .submission import StoryboardSubmission


def storyboard_tools() -> list[dict]:
    return [{"type": "function", "name": "submit_storyboard", "strict": True,
             "description": "Terminally submit the complete visual plan for the authorized Script. "
             "Supply only Agent-owned StoryboardSubmission fields. Runtime authenticates "
             "coverage, source identities, kinds, order and provenance and canonicalizes titles. "
             "This is a semantic handoff, not an external tool.",
             "parameters": strict_schema(StoryboardSubmission.model_json_schema())}]
