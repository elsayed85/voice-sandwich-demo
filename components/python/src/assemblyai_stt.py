"""
AssemblyAI Real-Time Streaming STT Transform

Python implementation that mirrors the TypeScript AssemblyAISTTTransform.
Connects to AssemblyAI's v3 WebSocket API for streaming speech-to-text.

Input: PCM 16-bit audio buffer (bytes)
Output: STT events (stt_chunk for partials, stt_output for final transcripts)
"""

import asyncio
import contextlib
import json
import os
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Optional
from urllib.parse import urlencode

import websockets
from websockets.client import WebSocketClientProtocol

from events import STTChunkEvent, STTEvent, STTOutputEvent


@dataclass
class EndpointingConfig:
    """
    Endpointing configuration for turn detection.
    Controls how the STT detects when the user has finished speaking.
    """

    end_of_turn_confidence_threshold: Optional[float] = None
    """
    Confidence threshold for semantic end-of-turn detection (0-1).
    Lower = more aggressive (faster responses), Higher = more conservative.
    Default: 0.4
    """

    min_end_of_turn_silence_when_confident: Optional[int] = None
    """
    Minimum silence duration (ms) when confident about end of turn.
    Lower = faster response when STT is confident the user finished.
    Default: 400
    """

    max_turn_silence: Optional[int] = None
    """
    Maximum silence duration (ms) before forcing end-of-turn.
    Acts as a fallback when semantic detection isn't confident.
    Default: 1280
    """


class AssemblyAISTT:
    def __init__(
        self,
        api_key: Optional[str] = None,
        sample_rate: int = 16000,
        format_turns: bool = True,
        endpointing: Optional[EndpointingConfig] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize AssemblyAI STT.

        Args:
            api_key: AssemblyAI API key (defaults to ASSEMBLYAI_API_KEY env var)
            sample_rate: Audio sample rate in Hz
            format_turns: Whether to format turns
            endpointing: Configuration for turn detection/endpointing behavior
            on_speech_start: Callback when speech is detected (partial transcript received).
                             Useful for implementing barge-in to interrupt TTS.
        """
        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            raise ValueError("AssemblyAI API key is required")

        self.sample_rate = sample_rate
        self.format_turns = format_turns
        self.endpointing = endpointing or EndpointingConfig()
        self.on_speech_start = on_speech_start
        self._ws: Optional[WebSocketClientProtocol] = None
        self._connection_signal = asyncio.Event()
        self._close_signal = asyncio.Event()
        # Track if we've already signaled speech start for current utterance
        self._speech_start_signaled = False

    async def receive_events(self) -> AsyncIterator[STTEvent]:
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
                            message_type = message.get("type")

                            if message_type == "Begin":
                                pass
                            elif message_type == "Turn":
                                transcript = message.get("transcript", "")
                                turn_is_formatted = message.get(
                                    "turn_is_formatted", False
                                )

                                if turn_is_formatted:
                                    if transcript:
                                        print(f'[AssemblyAI STT] Final transcript: "{transcript}"')
                                        yield STTOutputEvent.create(transcript)
                                    # Reset speech start flag for next utterance
                                    self._speech_start_signaled = False
                                else:
                                    yield STTChunkEvent.create(transcript)
                                    # Signal speech start for barge-in (only once per utterance)
                                    if (
                                        not self._speech_start_signaled
                                        and transcript
                                        and transcript.strip()
                                    ):
                                        self._speech_start_signaled = True
                                        if self.on_speech_start:
                                            self.on_speech_start()

                            elif message_type == "Termination":
                                print("[AssemblyAI STT] Received termination message")
                            else:
                                if "error" in message:
                                    print(f"AssemblyAISTT error: {message['error']}")
                                    break
                        except json.JSONDecodeError as e:
                            print(f"[DEBUG] AssemblyAISTT JSON decode error: {e}")
                            continue
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"[AssemblyAI STT] WebSocket closed: code={e.code}, reason={e.reason or 'none'}")

    async def send_audio(self, audio_chunk: bytes) -> None:
        try:
            ws = await self._ensure_connection()
            if ws.close_code is None:
                await ws.send(audio_chunk)
            else:
                print(f"[AssemblyAI STT] Cannot send audio, WebSocket closed with code: {ws.close_code}")
        except Exception as e:
            print(f"[AssemblyAI STT] Error sending audio: {e}")

    async def close(self) -> None:
        if self._ws and self._ws.close_code is None:
            await self._ws.close()
        self._ws = None
        self._close_signal.set()

    async def _ensure_connection(self) -> WebSocketClientProtocol:
        if self._close_signal.is_set():
            raise RuntimeError(
                "AssemblyAISTT tried establishing a connection after it was closed"
            )
        if self._ws and self._ws.close_code is None:
            return self._ws

        # Build query parameters
        query_params = {
            "sample_rate": self.sample_rate,
            "format_turns": str(self.format_turns).lower(),
        }

        # Add endpointing configuration if provided
        if self.endpointing.end_of_turn_confidence_threshold is not None:
            query_params["end_of_turn_confidence_threshold"] = str(
                self.endpointing.end_of_turn_confidence_threshold
            )
        if self.endpointing.min_end_of_turn_silence_when_confident is not None:
            query_params["min_end_of_turn_silence_when_confident"] = str(
                self.endpointing.min_end_of_turn_silence_when_confident
            )
        if self.endpointing.max_turn_silence is not None:
            query_params["max_turn_silence"] = str(self.endpointing.max_turn_silence)

        params = urlencode(query_params)
        url = f"wss://streaming.assemblyai.com/v3/ws?{params}"
        self._ws = await websockets.connect(
            url, additional_headers={"Authorization": self.api_key}
        )

        self._connection_signal.set()
        return self._ws
