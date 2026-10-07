"""Deterministic narration execution from exact immutable artifact references."""
import hashlib
import json

from history_studio.models import (
    ArtifactReference, GenerationMetadata, MediaType, NarrationAsset, ScriptPackage, StoryboardPackage,
)
from history_studio.storage import ArtifactStore, MediaStore

from .audio import measure_wav_duration
from .tts import TTSProvider, TTSResult


def generate_narration_assets(*, storyboard_input_ref: ArtifactReference,
                              artifact_store: ArtifactStore, provider: TTSProvider,
                              media_store: MediaStore) -> tuple[NarrationAsset, ...]:
    """Load exact Storyboard -> exact Script; synthesize represented segments once.

    Caller supplies the approved Storyboard reference. This service authenticates
    stored source lineage, not human approval; P7-D will own that workflow boundary.
    No caller-supplied Script payload, latest lookup, semantic rewrite or state write.
    All input validation happens before TTS. A later execution failure raises without
    a partial result; earlier immutable binary publications can remain as orphans.
    Retries synthesize again and accept identical bytes only. Regeneration/version
    policy and adoption of orphan assets remain future workflow responsibilities.
    """
    reference = ArtifactReference.model_validate(storyboard_input_ref.model_dump(mode="json"))
    if reference.artifact_type != "storyboard":
        raise ValueError("Narration requires an exact storyboard reference")
    if reference.project_id != artifact_store.project_dir.name:
        raise ValueError("Storyboard reference project must match artifact directory")
    storyboard = artifact_store.load("storyboard", reference.version, StoryboardPackage)
    storyboard = StoryboardPackage.model_validate(storyboard.model_dump(mode="json"))
    script_ref = storyboard.script_input_ref
    if script_ref.project_id != reference.project_id:
        raise ValueError("Storyboard Script provenance belongs to a different project")
    script = artifact_store.load("script", script_ref.version, ScriptPackage)
    script = ScriptPackage.model_validate(script.model_dump(mode="json"))
    if script.story_input_ref.project_id != reference.project_id:
        raise ValueError("Script provenance belongs to a different project")
    segments = [segment for section in script.sections for segment in section.segments]
    required = {shot.source_segment_id for section in storyboard.sections for shot in section.shots}
    unknown = required - {segment.segment_id for segment in segments}
    if unknown:
        raise ValueError("Unknown Storyboard source segments: " + ", ".join(sorted(unknown)))
    results = []
    for segment in segments:
        if segment.segment_id not in required:
            continue
        # Digest keeps IDs within Identifier's length limit and avoids filesystem
        # case collisions. Exact Storyboard/Script versions isolate immutable runs.
        identity = json.dumps([reference.model_dump(mode="json"), script_ref.model_dump(mode="json"),
                               segment.segment_id], sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        result = provider.synthesize(text=segment.narration)
        if not isinstance(result, TTSResult) or result.audio_format != "wav":
            raise ValueError("TTS provider must return a WAV TTSResult")
        metadata = GenerationMetadata(provider=result.provider, model=result.model)
        # Reject invalid audio before publication; final duration comes from an
        # integrity-checked read of the exact persisted artifact, not the provider.
        measure_wav_duration(result.audio_bytes)
        asset = media_store.save_bytes(asset_id=f"nar-{digest}",
            relative_path=f"audio/storyboard-v{reference.version}/{digest}.wav",
            media_type=MediaType.AUDIO, data=result.audio_bytes)
        duration = measure_wav_duration(media_store.read_bytes(asset))
        results.append(NarrationAsset(segment_id=segment.segment_id, asset=asset,
                                      duration_seconds=duration, generation_metadata=metadata))
    return tuple(results)
