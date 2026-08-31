"""Amazon Bedrock provider implementation using Anthropic's Bedrock client."""

from typing import Optional

from anthropic import AsyncAnthropicBedrock

from ..core.client import ModelClient
from ..models.anthropic import supports_structured_outputs
from .anthropic_client import AnthropicClient

# Claude models Bedrock serves through its Messages-API endpoint (Opus 4.7 and
# later). That endpoint does not offer structured outputs, unlike the legacy
# ARN-versioned integration that serves Opus 4.6 and earlier. Spelled as API
# identifiers so a match works against any Bedrock ID shape.
_NO_STRUCTURED_OUTPUTS_ON_BEDROCK = (
    "claude-fable-5",
    "claude-mythos",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
)


class BedrockClient(AnthropicClient):
    """
    Amazon Bedrock provider client for Claude models.

    Leverages Anthropic's built-in Bedrock support, which provides the same
    .messages.create() and .messages.stream() API as the regular Anthropic client.
    This means we can reuse all message conversion, tool calling, and streaming
    logic from AnthropicClient, including native structured outputs support.
    """

    def __init__(
        self,
        model: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        region_name: str,
        aws_session_token: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Bedrock client with AWS credentials.

        Args:
            model: Bedrock model ID. The shape depends on which Bedrock
                   integration serves the model: Opus 4.6 and earlier need a
                   cross-region inference profile ID (region prefix, e.g.
                   "us.anthropic.claude-sonnet-4-6") because on-demand
                   throughput is refused on their base IDs, while Opus 4.7 and
                   later are served by the Messages-API endpoint and take the
                   plain provider-prefixed ID ("anthropic.claude-opus-5"),
                   which has no inference profile to name.
            aws_access_key_id: AWS Access Key ID
            aws_secret_access_key: AWS Secret Access Key
            region_name: AWS region (e.g., "us-east-1", "us-west-2")
            aws_session_token: Optional AWS session token for temporary credentials
            **kwargs: Additional arguments passed to parent ModelClient
        """
        ModelClient.__init__(
            self,
            model=model,
            api_key=None,
            **kwargs
        )

        self.client = AsyncAnthropicBedrock(
            aws_access_key=aws_access_key_id,
            aws_secret_key=aws_secret_access_key,
            aws_region=region_name,
            aws_session_token=aws_session_token,
        )

        self.provider_name = "bedrock"
        self._tool_name_mapping = {}

    def _supports_structured_outputs(self) -> bool:
        """
        Check if the current Bedrock model supports native structured outputs.

        Support here is a property of the PLATFORM as well as the model, so the
        first-party answer is not reusable on its own. Bedrock's legacy
        ARN-versioned integration (Opus 4.6 and earlier: Opus 4.6, Sonnet 4.6,
        Sonnet 4.5, Opus 4.5, Haiku 4.5) supports structured outputs; the newer
        Messages-API endpoint that serves Opus 4.7 and later — Fable 5, Opus 5,
        Opus 4.8, Opus 4.7, Sonnet 5 — does not, and a request carrying the
        format is rejected there even though the same model accepts it on the
        Claude API. So a model must clear BOTH gates.

        Bedrock model IDs are matched by substring because they carry a
        provider prefix and, on the legacy path, a regional inference-profile
        prefix and version suffix ("us.anthropic.claude-sonnet-4-6").
        """
        model_lower = (self.model or "").lower()
        if any(name in model_lower for name in _NO_STRUCTURED_OUTPUTS_ON_BEDROCK):
            return False
        return supports_structured_outputs(self.model)

    def _supports_native_mcp(self) -> bool:
        """
        Check if Bedrock supports native MCP.

        Native MCP (the mcp-client beta) is an Anthropic API-specific feature
        that allows server-side MCP execution. This feature is NOT supported by
        Amazon Bedrock - attempting to use it results in:
        "extraneous key [mcp_servers] is not permitted"

        Returns:
            False - Bedrock does not support native MCP
        """
        return False

    # All other methods (achat, astream_chat, convert_schema_to_provider_format,
    # convert_message_to_provider_format, _prepare_messages, etc.) are inherited
    # from AnthropicClient and work as-is since Bedrock uses the same API!
