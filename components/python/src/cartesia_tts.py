"""
Cartesia Text-to-Speech Streaming

Python implementation of Cartesia's Sonic streaming TTS API.
Converts text to PCM audio in real-time using WebSocket streaming.

Input: Text strings
Output: TTS events (tts_chunk for audio chunks)
"""

import asyncio
import base64
import contextlib
import json
import os
import time
from typing import AsyncIterator, Callable, Literal, Optional

import websockets
from websockets.client import WebSocketClientProtocol

from events import TTSChunkEvent, TTSEndEvent


class CartesiaTTS:
    _ws: Optional[WebSocketClientProtocol]
    _connection_signal: asyncio.Event
    _close_signal: asyncio.Event
    _is_interrupted: bool

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "f6ff7c0c-e396-40a9-a70b-f7607edb6937",
        model_id: str = "sonic-3",
        sample_rate: int = 24000,
        encoding: Literal[
            "pcm_s16le", "pcm_f32le", "pcm_mulaw", "pcm_alaw"
        ] = "pcm_s16le",
        language: str = "en",
        cartesia_version: str = "2025-04-16",
        on_interrupt: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize Cartesia TTS.

        Args:
            api_key: Cartesia API key (defaults to CARTESIA_API_KEY env var)
            voice_id: Cartesia voice ID
            model_id: Cartesia model ID
            sample_rate: Audio sample rate in Hz
            encoding: Audio encoding format
            language: Language code
            cartesia_version: API version
            on_interrupt: Callback called when TTS output is interrupted (for barge-in).
        """
        self.api_key = api_key or os.getenv("CARTESIA_API_KEY")
        if not self.api_key:
            raise ValueError("Cartesia API key is required")

        self.voice_id = voice_id
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.language = language
        self.cartesia_version = cartesia_version
        self.on_interrupt = on_interrupt
        self._ws = None
        self._connection_signal = asyncio.Event()
        self._close_signal = asyncio.Event()
        self._context_counter = 0
        self._is_interrupted = False

    def _generate_context_id(self) -> str:
        """
        Generate a valid context_id for Cartesia.
        Context IDs must only contain alphanumeric characters, underscores, and hyphens.
        """
        timestamp = int(time.time() * 1000)
        counter = self._context_counter
        self._context_counter += 1
        return f"ctx_{timestamp}_{counter}"

    async def interrupt(self) -> None:
        """
        Interrupt the current TTS output (for barge-in support).
        Closes the connection to stop audio generation and clears any pending audio.
        """
        if self._is_interrupted:
            return

        print("[CartesiaTTS] Interrupted by user (barge-in)")
        self._is_interrupted = True

        # Close connection to stop audio generation
        if self._ws and self._ws.close_code is None:
            try:
                await self._ws.close()
            except Exception:
                pass  # Ignore close errors
            self._ws = None

        # Call the on_interrupt callback
        if self.on_interrupt:
            self.on_interrupt()

        # Reset interrupted state after a brief delay to allow new input
        await asyncio.sleep(0.1)
        self._is_interrupted = False

    async def send_text(self, text: Optional[str]) -> None:
        if text is None:
            return

        if not text.strip():
            return

        ws = await self._ensure_connection()

        payload = {
            "model_id": self.model_id,
            "transcript": text,
            "voice": {
                "mode": "id",
                "id": self.voice_id,
            },
            "output_format": {
                "container": "raw",
                "encoding": self.encoding,
                "sample_rate": self.sample_rate,
            },
            "language": self.language,
            "context_id": self._generate_context_id(),
        }
        await ws.send(json.dumps(payload))

    async def receive_events(self) -> AsyncIterator[TTSChunkEvent | TTSEndEvent]:
        while not self._close_signal.is_set():
            _, pending = await asyncio.wait(
                [
                    asyncio.create_task(self._close_signal.wait()),
                    asyncio.create_task(self._connection_signal.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

            with contextlib.suppress(asyncio.CancelledError):
                for task in pending:
                    task.cancel()

            if self._close_signal.is_set():
                break

            if self._ws and self._ws.close_code is None:
                self._connection_signal.clear()
                try:
                    async for raw_message in self._ws:
                        try:
                            message = json.loads(raw_message)
                            if "data" in message and message["data"] is not None:
                                audio_chunk = base64.b64decode(message["data"])
                                if audio_chunk:
                                    yield TTSChunkEvent.create(audio_chunk)
                            # Emit tts_end when Cartesia signals synthesis is complete
                            if message.get("done") or message.get("type") == "done":
                                yield TTSEndEvent.create()
                                break
                            if "error" in message and message["error"]:
                                print(f"[DEBUG] Cartesia error: {message['error']}")
                                break
                        except json.JSONDecodeError as e:
                            print(f"[DEBUG] Cartesia JSON decode error: {e}")
                            continue
                except websockets.exceptions.ConnectionClosed:
                    print("Cartesia: WebSocket connection closed")
                finally:
                    if self._ws and self._ws.close_code is None:
                        await self._ws.close()
                    self._ws = None

    async def close(self) -> None:
        if self._ws and self._ws.close_code is None:
            await self._ws.close()
        self._ws = None
        self._close_signal.set()

    async def _ensure_connection(self) -> WebSocketClientProtocol:
        if self._close_signal.is_set():
            raise RuntimeError(
                "CartesiaTTS tried establishing a connection after it was closed"
            )
        if self._ws and self._ws.close_code is None:
            return self._ws

        url = (
            f"wss://api.cartesia.ai/tts/websocket"
            f"?api_key={self.api_key}&cartesia_version={self.cartesia_version}"
        )
        self._ws = await websockets.connect(url)

        self._connection_signal.set()
        return self._ws
