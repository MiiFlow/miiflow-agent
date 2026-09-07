"""Every Anthropic image source, including tool results, must be validated."""

import asyncio
import base64
import threading
from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from miiflow_agent.core.message import ImageBlock, Message, TextBlock
from miiflow_agent.providers.anthropic_client import AnthropicClient
from miiflow_agent.providers.sdk_client_cache import sdk_client_scope
from miiflow_agent.utils.image import IMAGE_MAX_HEIGHT, IMAGE_MAX_WIDTH


@pytest.mark.parametrize("tool_result", [False, True])
@pytest.mark.parametrize("remote_url", [False, True])
def test_all_image_sources_are_resized(tool_result, remote_url):
    buf = BytesIO()
    Image.new("RGB", (100, 9001)).save(buf, format="PNG")
    raw = buf.getvalue()
    url = "https://example.com/screenshot.png" if remote_url else (
        "data:image/png;base64," + base64.b64encode(raw).decode()
    )
    content = [TextBlock(text="Screenshot"), ImageBlock(image_url=url)]
    message = Message.tool(content, "call_1") if tool_result else Message.user(content)
    client = object.__new__(AnthropicClient)
    with patch("miiflow_agent.utils.image.httpx.Client") as http:
        response = http.return_value.__enter__.return_value.get.return_value
        response.content = raw
        response.headers = {"content-type": "image/png"}
        result = client.convert_message_to_provider_format(message)
    blocks = result["content"][0]["content"] if tool_result else result["content"]
    assert blocks[0] == {"type": "text", "text": "Screenshot"}
    source = blocks[1]["source"]
    assert source["type"] == "base64"
    with Image.open(BytesIO(base64.b64decode(source["data"]))) as decoded:
        assert decoded.width <= IMAGE_MAX_WIDTH
        assert decoded.height <= IMAGE_MAX_HEIGHT


@pytest.mark.parametrize("url", [
    "data:image/png;base64,%%%",
    "data:image/png;base64," + base64.b64encode(b"not an image").decode(),
    "https://example.com/missing.png",
])
@pytest.mark.parametrize("tool_result", [False, True])
def test_invalid_images_preserve_other_content_without_url_fallback(url, tool_result):
    content = [TextBlock(text="Useful result"), ImageBlock(image_url=url)]
    message = Message.tool(content, "call_1") if tool_result else Message.user(content)
    with patch("miiflow_agent.utils.image.httpx.Client", side_effect=OSError("unavailable")):
        result = object.__new__(AnthropicClient).convert_message_to_provider_format(message)
    blocks = result["content"][0]["content"] if tool_result else result["content"]
    assert blocks[0]["text"] == "Useful result"
    assert blocks[1]["type"] == "text"
    assert "unavailable" in blocks[1]["text"]
    assert url not in str(blocks)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_image_validation_does_not_block_the_asgi_loop(
    streaming, mock_anthropic_response, mock_anthropic_stream_chunks,
):
    from miiflow_agent.utils.image import resize_image_for_api

    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    started, release = asyncio.Event(), threading.Event()
    worker_threads = []
    buf = BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def validate(*args, **kwargs):
        worker_threads.append(threading.get_ident())
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5), "image processing blocked the event loop"
        return resize_image_for_api(*args, **kwargs)

    async def chunks():
        for chunk in mock_anthropic_stream_chunks:
            yield chunk

    async with sdk_client_scope():
        client = AnthropicClient(model="claude-3-haiku-20240307", api_key="test-key")
        with (
            patch("miiflow_agent.utils.image.resize_image_for_api", side_effect=validate),
            patch.object(client.client.messages, "create", new_callable=AsyncMock) as create,
        ):
            create.return_value = chunks() if streaming else mock_anthropic_response

            async def run():
                messages = [Message.user([ImageBlock(image_url=url)])]
                if streaming:
                    return [chunk async for chunk in client.astream_chat(messages)]
                return await client.achat(messages)

            task = asyncio.create_task(run())
            try:
                await asyncio.wait_for(started.wait(), 5)
                assert worker_threads and all(t != loop_thread for t in worker_threads)
                assert not task.done()
            finally:
                release.set()
                await task
            create.assert_awaited_once()
