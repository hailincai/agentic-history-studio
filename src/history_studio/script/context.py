"""Exact Story snapshot projection, without generation or workflow authorization."""
from history_studio.models.artifact_reference import ArtifactReference
from history_studio.models.script_context import ScriptContext
from history_studio.models.story_package import StoryPackage
from history_studio.storage.artifact_store import ArtifactStore


def build_script_context(store: ArtifactStore, *,
                         story_input_ref: ArtifactReference) -> ScriptContext:
    """Project only the exact Story reference supplied from approved_story by Runtime.

    Approval is a caller precondition for future workflow integration. Store loading
    revalidates all nested Story contracts; no latest lookup, upstream artifact load,
    semantic repair or output-side Script creation occurs. JSON-shaped reconstruction
    detaches every nested value from the loaded artifact and caller's reference.
    """
    # Read raw fields before serialization can coerce an unvalidated model_copy.
    reference = ArtifactReference(project_id=story_input_ref.project_id,
                                  artifact_type=story_input_ref.artifact_type,
                                  version=story_input_ref.version)
    if reference.artifact_type != "story":
        raise ValueError("Script input must reference a story artifact")
    if store.project_dir.name != reference.project_id:
        raise ValueError("ArtifactStore project must match story reference")
    story = store.load("story", reference.version, StoryPackage)
    if story.verification_input_ref.project_id != reference.project_id:
        raise ValueError("Story provenance project must match exact story reference")
    return ScriptContext.model_validate(story.plan.model_dump(mode="json") | {
        "story_input_ref": reference.model_dump(mode="json"),
    })
