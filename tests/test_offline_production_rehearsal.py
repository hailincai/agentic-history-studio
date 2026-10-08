"""P9-E0: synthetic evidence, fake external providers, actual runtime and FFmpeg.

Fixture approvals exercise Human Gate contracts; they are not human authentication
or approval of real-world historical claims. Audio is a tone, not spoken TTS.
"""
import hashlib
import io
import json
import math
import shutil
import struct
import subprocess
import wave
from datetime import datetime, timezone
from fractions import Fraction

import pytest

from history_studio.assembly import FFmpegRenderer
from history_studio.assembly.rendering import DURATION_TOLERANCE_SECONDS
from history_studio.budget import ProjectBudget
from history_studio.media.recovery import AssetRecovery, RecoveryJournal
from history_studio.media.tts import TTSResult
from history_studio.model_io import ModelResponse, NativeToolCall
from history_studio.models import ProjectConfig, StoryPackage, ScriptPackage, StoryboardPackage, AssemblyPackage
from history_studio.models.research_package import ResearchRunStatus
from history_studio.research.agent import ResearchAgent
from history_studio.research.boundaries import ToolObservation
from history_studio.research.spans import make_spans
from history_studio.research.web_tools import source_reference
from history_studio.storage import ArtifactStore, MediaStore
from history_studio.storage.artifact_store import write_json
from history_studio.verification import FactCheckingRunner
from history_studio.workflow import RuntimeState, ProjectState as S, ApprovalRecord
from history_studio.workflow.assembly import AssemblyWorkflow
from history_studio.workflow.fact_checking import FactCheckingWorkflow
from history_studio.workflow.fact_review import apply_fact_review, load_fact_review
from history_studio.workflow.story import StoryWorkflow
from history_studio.workflow.story_review import apply_story_review, load_story_review
from history_studio.workflow.script import ScriptWorkflow
from history_studio.workflow.script_review import apply_script_review, load_script_review
from history_studio.workflow.storyboard import StoryboardWorkflow
from history_studio.workflow.storyboard_review import apply_storyboard_review, load_storyboard_review
from history_studio.workflow.media import MediaWorkflow, MediaProviders
from test_research_agent import FakeProvider, settings, action, update
from test_fact_checker_dispatch import native
from test_fact_checker_investigation import SequenceProvider, decision
from test_verification_evidence import TextTools
from test_verification_submission import payload, terminal
from test_visual_media import FakeImageProvider, png_bytes


SYNTHETIC = "仅用于离线演练的合成记录：李白生于701年；本记录不是历史证据。"
NARRATION = (
    "这是李白纪录片的离线制作演练。这里的记录和画面均为合成测试素材，不能作为真实历史证据。",
    "演练记录写着李白生于七〇一年。这句话仅用于检查来源绑定、中文旁白、字幕时序和视频组装流程。",
)


def persisted_state(store):
    return RuntimeState.model_validate_json((store.project_dir / ".runtime/state.json").read_text(encoding="utf-8"))


class SyntheticResearchTools:
    def __init__(self):
        self.source = source_reference("https://example.invalid/synthetic-li-bai", title="SYNTHETIC offline fixture")

    def search_reserve_cost(self, query):
        return 0

    def search_web(self, query):
        return ToolObservation(kind="search", sources=[self.source])

    def read_source(self, source, max_chars):
        assert source.source_id == self.source.source_id
        return ToolObservation(kind="source", sources=[source], source_id=source.source_id, text=SYNTHETIC)


class CreativeProvider:
    def __init__(self, stage, context, requests):
        self.stage, self.context, self.requests = stage, context, requests

    def decide(self, request):
        context = json.loads(request.input)
        assert context == self.context.model_dump(mode="json")
        brief = context["production_brief"]
        assert brief == dict(topic="李白", language="zh-CN", target_duration_seconds=45.0,
                             allowed_generation_methods=["STATIC_IMAGE"])
        self.requests[self.stage].append(request)
        if self.stage == "story":
            fact_id = context["eligible_facts"][0]["research_fact_id"]
            proposal = dict(title="李白：离线合成演练", narrative_thesis="检查制作流程，不建立历史结论", sections=[dict(
                section_id="opening", purpose="四十五秒合成制作演练", beats=[dict(beat_id="origin",
                    kind="historical", narrative_role="opening", summary=SYNTHETIC,
                    fact_proposals=[dict(research_fact_id=fact_id, use="AFFIRMATIVE")])])])
        elif self.stage == "script":
            section = context["sections"][0]
            beat = section["beats"][0]
            proposal = dict(title="李白：离线合成演练", sections=[dict(section_id=section["section_id"],
                title="合成材料说明", segments=[dict(segment_id=f"seg-{index}", kind="HISTORICAL", narration=text,
                    grounding=dict(story_beat_id=beat["beat_id"], research_fact_ids=[item["research_fact_id"] for item in beat["fact_refs"]]))
                    for index, text in enumerate(NARRATION, 1)])])
        else:
            proposal = dict(title="李白：离线合成演练", sections=[dict(section_id=section["section_id"], title=section["title"],
                shots=[dict(shot_id=f"shot-{segment['segment_id']}", kind=segment["kind"], source_segment_id=segment["segment_id"],
                    visual_description="合成山水，不声称真实人物肖像", generation_prompt="合成测试山水背景，无文字",
                    generation_method="STATIC_IMAGE", framing="WIDE", camera_motion="NONE", estimated_duration_seconds=22.5)
                    for segment in section["segments"]]) for section in context["sections"]])
        name = {"story": "submit_story", "script": "submit_script", "storyboard": "submit_storyboard"}[self.stage]
        assert [tool["name"] for tool in request.tools] == [name]
        return ModelResponse(status="completed", usage={}, tool_calls=[NativeToolCall(call_id=f"offline-{name}",
            name=name, arguments=json.dumps(proposal, ensure_ascii=False))])


class ToneTTS:
    model = "synthetic-45-second-tone"

    def __init__(self):
        self.texts = []

    def synthesize(self, *, text):
        self.texts.append(text)
        seconds = {NARRATION[0]: 22, NARRATION[1]: 23}[text]
        rate = 48000
        tone = b"".join(struct.pack("<h", round(1500 * math.sin(2 * math.pi * 440 * sample / rate))) for sample in range(rate))
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(rate)
            audio.writeframes(tone * seconds)
        return TTSResult(audio_bytes=buffer.getvalue(), provider="offline-tone", model=self.model)


def test_offline_li_bai_documentary_complete(tmp_path, monkeypatch):
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("Offline production rehearsal requires local ffmpeg and ffprobe")
    def forbidden(*args, **kwargs):
        pytest.fail("Offline rehearsal must not instantiate paid providers or discover latest artifacts")
    from openai import OpenAI, AsyncOpenAI
    import history_studio.media.openai_provider as paid_media
    monkeypatch.setattr(OpenAI, "__init__", forbidden)
    monkeypatch.setattr(AsyncOpenAI, "__init__", forbidden)
    for name in ("OpenAITTSProvider", "OpenAIImageProvider", "OpenAIVideoProvider"):
        monkeypatch.setattr(paid_media, name, forbidden)
    monkeypatch.setattr(ArtifactStore, "load_latest", forbidden)

    project = ProjectConfig(project_id="li_bai", topic="李白", research_scope="仅合成材料的离线制作演练",
        language="zh-CN", target_duration_minutes=0.75, output_width=1280, output_height=720,
        budget_usd=1, allowed_generation_methods=["STATIC_IMAGE"])
    store = ArtifactStore(tmp_path / project.project_id)
    write_json(store.project_dir / "project.json", project)
    write_json(store.project_dir / ".runtime/state.json", RuntimeState())
    budget = ProjectBudget(store.project_dir, stage="offline-rehearsal")
    budget.initialize_new()
    budget_before = budget.path.read_bytes()
    states = ["CREATED"]
    tools = SyntheticResearchTools()
    checkpoint = update()
    fact = checkpoint.arguments["facts"][0]
    fact.update(claim="合成演练记录称李白生于701年。", research_notes="Synthetic offline evidence only", evidence=[dict(
        source_id=tools.source.source_id, span_id=next(iter(make_spans(tools.source.source_id, SYNTHETIC).spans)))])
    goal = checkpoint.arguments["plan"]["gaps"][0]
    goal["question"] = "合成演练记录如何描述李白的出生？"
    research = ResearchAgent(FakeProvider([action("search_web", query="离线李白合成记录"),
        action("read_source", source_id=tools.source.source_id), checkpoint], cost=0), tools, settings()).run(project, store)
    assert research.progress.status == ResearchRunStatus.COMPLETE
    states.append(persisted_state(store).current_state.value)
    research_ref = persisted_state(store).artifacts.research

    verifier_requests = []
    def verification_provider(context):
        assert context.research_input_ref == research_ref
        submission = payload("VERIFIED", source_id="SRC-new")
        submission.update(verification_evidence=[dict(source_id="SRC-new", span_id=next(iter(make_spans("SRC-new", SYNTHETIC).spans)))],
            independence_note="Both records are synthetic fixtures; no real-world corroboration", rationale="Synthetic protocol test only")
        provider = SequenceProvider([decision(native("search_web", '{"query":"offline synthetic record"}')),
            decision(native("read_source", '{"source_id":"SRC-new"}')), decision(terminal(submission))])
        verifier_requests.append(provider)
        return provider
    verification = FactCheckingWorkflow(FactCheckingRunner(provider_factory=verification_provider,
        tools_factory=lambda context: TextTools(SYNTHETIC))).run(project, store)
    assert verification.state.current_state == S.WAITING_FACT_APPROVAL
    states.append(verification.state.current_state.value)
    assert len(verifier_requests[0].requests) == 3

    def gate(stage, load_review, apply_review):
        state = persisted_state(store)
        candidate = getattr(state.artifacts, "verification" if stage == "facts" else stage)
        assert load_review(state, store) is not None
        record = ApprovalRecord(project_id=project.project_id, stage=stage, artifact_type=candidate.artifact_type,
            artifact_version=candidate.version, decision="APPROVED", decided_by="Offline fixture reviewer (simulated)",
            decision_source="human", decided_at=datetime.now(timezone.utc), feedback="Synthetic rehearsal only; not real-world historical approval")
        approved = apply_review(store.project_dir, record)
        assert getattr(approved.artifacts, "approved_verification" if stage == "facts" else f"approved_{stage}") == candidate
        states.append(approved.current_state.value)

    gate("facts", load_fact_review, apply_fact_review)
    creative_requests = {stage: [] for stage in ("story", "script", "storyboard")}
    for stage, workflow_type, load_review, apply_review in (
        ("story", StoryWorkflow, load_story_review, apply_story_review),
        ("script", ScriptWorkflow, load_script_review, apply_script_review),
        ("storyboard", StoryboardWorkflow, load_storyboard_review, apply_storyboard_review)):
        outcome = workflow_type(provider_factory=lambda context, stage=stage: CreativeProvider(stage, context, creative_requests)).run(project, store)
        assert outcome.state.current_state.value == f"WAITING_{stage.upper()}_APPROVAL"
        states.append(outcome.state.current_state.value)
        assert len(creative_requests[stage]) == 1
        gate(stage, load_review, apply_review)
    approved = persisted_state(store).artifacts
    story = store.load("story", approved.approved_story.version, StoryPackage)
    script = store.load("script", approved.approved_script.version, ScriptPackage)
    board = store.load("storyboard", approved.approved_storyboard.version, StoryboardPackage)
    assert story.verification_input_ref == approved.approved_verification
    assert script.story_input_ref == approved.approved_story and board.script_input_ref == approved.approved_script
    assert {shot.generation_method.value for section in board.sections for shot in section.shots} == {"STATIC_IMAGE"}
    assert {shot.source_segment_id for section in board.sections for shot in section.shots} == {"seg-1", "seg-2"}
    assert store.list_versions("approvals") == [1, 2, 3, 4]

    tts, image = ToneTTS(), FakeImageProvider([], data=png_bytes())
    providers = MediaProviders(tts, image, recovery_identity={"fixture": "tone-and-png-v1", "seconds": [22, 23]})
    media_runner = MediaWorkflow(provider_factory=lambda board: providers)
    perform = AssetRecovery.perform
    def local_interruption(self, **kwargs):
        if kwargs["key"] == "narration:seg-2":
            raise OSError("safe local interruption BEFORE second dispatch intent")
        return perform(self, **kwargs)
    monkeypatch.setattr(AssetRecovery, "perform", local_interruption)
    first = media_runner.run(project, store)
    assert first.state.current_state == S.FAILED and first.state.artifacts.media is None
    assert tts.texts == [NARRATION[0]]
    monkeypatch.setattr(AssetRecovery, "perform", perform)
    media = media_runner.run(project, store)
    assert media.state.current_state == S.ASSEMBLING and tts.texts == list(NARRATION)
    assert sum(asset.duration_seconds for asset in media.package.narration_assets) == 45
    journal = RecoveryJournal.model_validate_json((store.project_dir / ".runtime/media_recovery.json").read_text(encoding="utf-8"))
    assert len(journal.entries) == 4 and {entry.state for entry in journal.entries.values()} == {"COMPLETED"}
    assert journal.scope.script_ref == approved.approved_script and journal.scope.storyboard_ref == approved.approved_storyboard
    states.extend(["FAILED (safe local pre-dispatch interruption)", "ASSEMBLING"])
    calls = len(tts.texts), len(image.prompts)
    assert MediaWorkflow(provider_factory=forbidden).run(project, ArtifactStore(store.project_dir)).package == media.package
    assert calls == (len(tts.texts), len(image.prompts))

    render_root = tmp_path / "rendered"
    render_calls = []
    def renderer():
        render_calls.append(True)
        return FFmpegRenderer(render_root, ffmpeg=ffmpeg, ffprobe=ffprobe, timeout_seconds=300)
    assembly = AssemblyWorkflow(render_root=render_root, renderer_factory=renderer).run(project, store)
    assert assembly.state.current_state == S.COMPLETE, assembly
    assert len(render_calls) == 1 and assembly.package.media_input_ref == media.state.artifacts.media
    bound = store.load("assembly", assembly.state.artifacts.assembly.version, AssemblyPackage)
    assert bound == assembly.package
    final_store = MediaStore(render_root)
    video_path, srt_path = render_root / bound.final_video.relative_path, render_root / bound.subtitles.relative_path
    final_store.verify(bound.final_video)
    expected_srt = f"1\n00:00:00,000 --> 00:00:22,000\n{NARRATION[0]}\n\n2\n00:00:22,000 --> 00:00:45,000\n{NARRATION[1]}\n\n".encode("utf-8")
    assert final_store.read_bytes(bound.subtitles) == expected_srt
    probe = subprocess.run([ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe", "-show_streams", "-show_format", "-of", "json", str(video_path)],
                           capture_output=True, timeout=60, check=True)
    metadata = json.loads(probe.stdout)
    video, audio = next(s for s in metadata["streams"] if s["codec_type"] == "video"), next(s for s in metadata["streams"] if s["codec_type"] == "audio")
    assert len(metadata["streams"]) == 2 and video["codec_name"] == "h264" and audio["codec_name"] == "aac"
    assert (video["width"], video["height"], Fraction(video["avg_frame_rate"])) == (1280, 720, 30)
    duration = float(metadata["format"]["duration"])
    assert 30 <= duration <= 60 and abs(duration - 45) <= DURATION_TOLERANCE_SECONDS
    subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-protocol_whitelist", "file,pipe", "-i", str(video_path),
                    "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"], capture_output=True, timeout=120, check=True)
    before = (store.project_dir / ".runtime/state.json").read_bytes()
    assert AssemblyWorkflow(render_root=render_root, renderer_factory=forbidden).run(project, ArtifactStore(store.project_dir)).package == bound
    assert (store.project_dir / ".runtime/state.json").read_bytes() == before
    assert len(render_calls) == 1 and store.list_versions("assembly") == [assembly.state.artifacts.assembly.version]
    assert assembly.state.artifacts.model_dump(exclude={"media", "assembly"}) == approved.model_dump(exclude={"media", "assembly"})
    assert budget.path.read_bytes() == budget_before and budget.snapshot()["committed_usd"] == 0 and budget.snapshot()["outstanding_usd"] == 0
    states.append("COMPLETE")
    report = dict(synthetic_only=True, states=states, lineage=assembly.state.artifacts.model_dump(mode="json"),
        video=str(video_path), srt=str(srt_path), codec="H.264/AAC", size="1280x720", fps=30, duration_seconds=duration,
        video_sha256=hashlib.sha256(video_path.read_bytes()).hexdigest(), srt_sha256=hashlib.sha256(srt_path.read_bytes()).hexdigest(),
        budget_committed_usd=0, budget_outstanding_usd=0, media_dispatches=len(journal.entries), render_calls=len(render_calls))
    (tmp_path / "rehearsal-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
