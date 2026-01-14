"""Amazon Bedrock provider implementation using Anthropic's Bedrock client."""

from typing import Optional

from anthropic import AsyncAnthropicBedrock

from ..core.client import ModelClient
from ..models.anthropic import supports_structured_outputs
from .anthropic_client import AnthropicClient


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
            model: Bedrock inference profile ID (e.g., "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
                   IMPORTANT: Must use inference profile IDs (with region prefix like "us.")
                   for on-demand throughput, not base model IDs.
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

        Bedrock model IDs follow pattern: "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        We check the base model name against supported models.
        """
        return supports_structured_outputs(self.model)

    def _supports_native_mcp(self) -> bool:
        """
        Check if Bedrock supports native MCP.

        Native MCP (mcp-client-2025-04-04 beta) is an Anthropic API-specific feature
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
