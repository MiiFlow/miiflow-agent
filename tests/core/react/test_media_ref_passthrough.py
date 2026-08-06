"""Tests for schema-declared media-ref passthrough and the extracted resolver.

Which parameters consume symbolic ``media_ref:`` values used to be a table of
application tool names inside the orchestrator; tools now declare it on their
own schema via ``ParameterSchema(media_ref_passthrough=True)``.
"""

from miiflow_agent.core.react.media_refs import (
    declared_passthrough_params,
    resolve_media_refs,
)
from miiflow_agent.core.tools.schemas import ParameterSchema, ToolSchema
from miiflow_agent.core.tools.types import ParameterType, ToolType

MEDIA_ID = "0f8fad5b-d9cb-469f-a165-70867728950e"
STORE = {MEDIA_ID: "https://cdn.example/img.png"}


def _schema(**param_flags):
    params = {
        name: ParameterSchema(
            name=name,
            type=ParameterType.STRING,
            description=name,
            media_ref_passthrough=flag,
        )
        for name, flag in param_flags.items()
    }
    return ToolSchema(
        name="t", description="t", tool_type=ToolType.FUNCTION, parameters=params
    )


class TestDeclaredPassthrough:
    def test_reads_flags_from_schema(self):
        schema = _schema(image_ref=True, prompt=False)
        assert declared_passthrough_params(schema) == {"image_ref"}

    def test_empty_for_flagless_schema(self):
        assert declared_passthrough_params(_schema(prompt=False)) == set()
        assert declared_passthrough_params(None) == set()


class TestResolveMediaRefs:
    def test_explicit_ref_resolves(self):
        out = resolve_media_refs({"image": f"media_ref:{MEDIA_ID}"}, STORE)
        assert out["image"] == STORE[MEDIA_ID]

    def test_passthrough_param_is_untouched(self):
        ref = f"media_ref:{MEDIA_ID}"
        out = resolve_media_refs(
            {"image": ref}, STORE, passthrough_params={"image"}
        )
        assert out["image"] == ref

    def test_hallucinated_sandbox_path_resolves_by_uuid(self):
        out = resolve_media_refs({"image": f"/mnt/data/{MEDIA_ID}.png"}, STORE)
        assert out["image"] == STORE[MEDIA_ID]

    def test_lone_media_file_path_fallback(self):
        out = resolve_media_refs({"image": "/tmp/whatever.png"}, STORE)
        assert out["image"] == STORE[MEDIA_ID]

    def test_urls_and_non_strings_pass_through(self):
        inputs = {"image": "https://other.example/x.png", "count": 3}
        out = resolve_media_refs(inputs, STORE)
        assert out == inputs

    def test_empty_store_is_identity(self):
        inputs = {"image": f"media_ref:{MEDIA_ID}"}
        assert resolve_media_refs(inputs, {}) is inputs
