"""Exact Script snapshot projection, without generation or workflow authorization."""
from history_studio.models.artifact_reference import ArtifactReference
from history_studio.models.script_package import ScriptPackage
from history_studio.models.visual_director_context import VisualDirectorContext
from history_studio.storage.artifact_store import ArtifactStore


def build_visual_director_context(store: ArtifactStore, *,
                                  script_input_ref: ArtifactReference) -> VisualDirectorContext:
    """Project only the exact reference supplied from approved_script by Runtime.

    No latest discovery, upstream loading, or approval inference occurs. Store
    loading revalidates ScriptPackage; JSON-shaped reconstruction detaches nested
    values and the caller's reference. Script's upstream project is checked only
    as identity metadata, without loading or exposing that upstream artifact.
    """
    # Preserve raw types so an unvalidated model_copy cannot coerce its version.
    reference = ArtifactReference(project_id=script_input_ref.project_id,
                                  artifact_type=script_input_ref.artifact_type,
                                  version=script_input_ref.version)
    if reference.artifact_type != "script":
        raise ValueError("Visual Director input must reference a script artifact")
    if store.project_dir.name != reference.project_id:
        raise ValueError("ArtifactStore project must match script reference")
    script = store.load("script", reference.version, ScriptPackage)
    if script.story_input_ref.project_id != reference.project_id:
        raise ValueError("Script provenance project must match exact script reference")
    return VisualDirectorContext.model_validate(script.model_dump(
        mode="json", include={"title", "sections"}) | {
            "script_input_ref": reference.model_dump(mode="json"),
        })
