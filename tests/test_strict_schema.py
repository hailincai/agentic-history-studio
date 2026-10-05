from copy import deepcopy

import pytest

from history_studio.research.openai_provider import native_tools, strict_schema
from history_studio.script.submission import ScriptSubmission
from history_studio.story.submission import StorySubmission


def assert_contract_properties(original, normalized):
    """Strictification retains every property name and makes each required."""
    if isinstance(original, list):
        for first, second in zip(original, normalized):
            assert_contract_properties(first, second)
    elif isinstance(original, dict):
        if "properties" in original:
            assert set(normalized["properties"]) == set(original["properties"])
            assert normalized["required"] == list(original["properties"])
            assert normalized["additionalProperties"] is False
        for key, value in original.items():
            if key not in ("title", "default") and key in normalized:
                assert_contract_properties(value, normalized[key])


def test_annotations_removed_business_keyword_names_retained_nested_and_in_arrays():
    scalar = {"title": "Annotation", "default": "value", "type": "string", "description": "Keep guidance"}
    inner = {"title": "Inner", "type": "object", "properties": {"title": scalar}, "required": ["title"]}
    schema = {"title": "Root", "type": "object", "properties": {
        name: scalar for name in ("title", "default", "description", "examples", "ordinary")},
        "required": ["title", "default", "description", "examples", "ordinary"]}
    schema["properties"]["nested"] = {"type": "object", "properties": {"title": inner}}
    schema["properties"]["items"] = {"type": "array", "items": inner}
    before = deepcopy(schema)
    result = strict_schema(schema)
    assert "title" not in result
    assert result["properties"]["title"] == {"type": "string", "description": "Keep guidance"}
    assert set(result["properties"]) == set(schema["properties"])
    assert result["required"] == list(schema["properties"])
    nested = result["properties"]["nested"]["properties"]["title"]
    assert nested["required"] == ["title"] and "title" in nested["properties"]
    assert result["properties"]["items"]["items"]["required"] == ["title"]
    assert_contract_properties(schema, result)
    assert schema == before
    assert strict_schema(schema) == result


def test_named_definitions_and_ref_siblings_preserve_title_properties():
    schema = {"$defs": {"title": {"title": "Annotation", "type": "object", "properties": {
        "title": {"title": "Field annotation", "type": "string"}}}}, "type": "object",
        "properties": {"plain": {"$ref": "#/$defs/title"},
                       "expanded": {"$ref": "#/$defs/title", "description": "Local guidance"}}}
    result = strict_schema(schema)
    assert result["$defs"]["title"]["required"] == ["title"]
    assert "title" not in result["$defs"]["title"]
    assert result["properties"]["plain"] == {"$ref": "#/$defs/title"}
    expanded = result["properties"]["expanded"]
    assert expanded["description"] == "Local guidance"
    assert expanded["required"] == ["title"]
    assert expanded["properties"]["title"] == {"type": "string"}


@pytest.mark.parametrize("mapping", ["definitions", "patternProperties", "dependentSchemas"])
def test_other_schema_name_mappings_preserve_names(mapping):
    schema = {mapping: {"title": {"title": "Annotation", "type": "string"},
                        "default": {"type": "integer"}}}
    assert strict_schema(schema)[mapping] == {"title": {"type": "string"}, "default": {"type": "integer"}}


@pytest.mark.parametrize("contract,section_definition", [
    (ScriptSubmission, "ScriptSectionSubmission"), (StorySubmission, "StorySectionProposal"),
])
def test_actual_script_and_story_contract_compatibility(contract, section_definition):
    schema = contract.model_json_schema()
    before = deepcopy(schema)
    result = strict_schema(schema)
    assert "title" in result["properties"] and "title" in result["required"]
    assert "title" not in result["properties"]["title"]
    if contract is ScriptSubmission:
        nested = result["$defs"][section_definition]
        assert "title" in nested["properties"] and "title" in nested["required"]
    assert_contract_properties(schema, result)
    assert schema == before and strict_schema(schema) == result


def test_p3_native_tools_retain_complete_contract_shapes():
    from history_studio.research.openai_provider import action_contracts

    contracts = action_contracts()
    for tool in native_tools():
        assert tool["strict"] is True
        original = contracts[tool["name"]].model_json_schema()
        # The existing P3 native boundary intentionally omits legacy reader fields.
        if tool["name"] == "checkpoint_research":
            for legacy in ("sources", "time_period"):
                original["$defs"]["ResearchFactProposal"]["properties"].pop(legacy)
            for legacy in ("SourceReference", "SourceType"):
                original["$defs"].pop(legacy, None)
        assert_contract_properties(original, tool["parameters"])
