"""Terminal narrative handoff schema, never an external research capability."""
from history_studio.research.openai_provider import strict_schema
from .submission import StorySubmission


def story_tools() -> list[dict]:
    return [{"type": "function", "name": "submit_story", "strict": True,
             "description": "Terminally submit the narrative proposal. Supply only Agent-owned "
             "StorySubmission fields. Runtime authenticates facts, status, chronology and provenance. "
             "This is a semantic handoff, not an external tool.",
             "parameters": strict_schema(StorySubmission.model_json_schema())}]
