"""
Thinking Filler Transform

A transform that emits "filler" phrases (e.g., "Let me see...", "Hmm, one moment...")
when the upstream agent takes longer than a specified threshold to respond.
This creates a more natural, conversational experience for voice applications.
"""

import asyncio
import random
from typing import AsyncIterator, Callable, Optional

from events import AgentChunkEvent, AgentEndEvent, VoiceAgentEvent

DEFAULT_FILLER_PHRASES = [
    "Let me see here...",
    "Hmm, one moment...",
    "Ah, let me think...",
    "Just a second...",
    "Mhm, okay...",
    "Let me check that...",
]


async def thinking_filler_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
    threshold_ms: int = 1200,
    filler_phrases: Optional[list[str]] = None,
    enabled: bool = True,
    on_filler_emitted: Optional[Callable[[str], None]] = None,
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Voice Events → Voice Events (with Filler Phrases)

    This function adds "thinking filler" phrases when the agent takes longer
    than a threshold to respond. This creates a more natural, conversational
    experience for voice applications.

    When an stt_output event arrives, it starts a timer. If no agent_chunk
    event arrives within the threshold, it emits a filler phrase as an
    agent_chunk event. The timer is cancelled when agent_chunk arrives.

    Args:
        event_stream: An async iterator of upstream voice agent events
        threshold_ms: Time in milliseconds before emitting a filler phrase
        filler_phrases: Array of filler phrases to randomly choose from
        enabled: Whether the filler functionality is enabled
        on_filler_emitted: Callback when a filler phrase is emitted

    Yields:
        All upstream events plus potential filler phrases
    """
    if not enabled:
        async for event in event_stream:
            yield event
        return

    if filler_phrases is None:
        filler_phrases = DEFAULT_FILLER_PHRASES

    # State for tracking filler timer
    filler_task: Optional[asyncio.Task] = None
    filler_emitted_this_turn = False
    waiting_for_agent_response = False
    filler_queue: asyncio.Queue[VoiceAgentEvent] = asyncio.Queue()

    def pick_filler_phrase() -> str:
        return random.choice(filler_phrases)

    def cancel_filler_timer():
        nonlocal filler_task
        if filler_task and not filler_task.done():
            filler_task.cancel()
            filler_task = None

    async def emit_filler_after_delay():
        nonlocal filler_emitted_this_turn
        try:
            await asyncio.sleep(threshold_ms / 1000.0)

            if filler_emitted_this_turn or not waiting_for_agent_response:
                return

            phrase = pick_filler_phrase()
            print(f'[ThinkingFiller] Emitting filler: "{phrase}"')

            # Queue the filler events
            await filler_queue.put(AgentChunkEvent.create(phrase))
            await filler_queue.put(AgentEndEvent.create())

            filler_emitted_this_turn = True
            if on_filler_emitted:
                on_filler_emitted(phrase)

        except asyncio.CancelledError:
            pass

    try:
        async for event in event_stream:
            # When we receive stt_output, start the filler timer
            if event.type == "stt_output":
                filler_emitted_this_turn = False
                waiting_for_agent_response = True
                cancel_filler_timer()

                print(
                    f"[ThinkingFiller] STT output received, starting {threshold_ms}ms timer"
                )

                filler_task = asyncio.create_task(emit_filler_after_delay())

            # When we receive agent_chunk, cancel the filler timer
            if event.type == "agent_chunk":
                waiting_for_agent_response = False
                cancel_filler_timer()

            # Yield any queued filler events first
            while not filler_queue.empty():
                yield filler_queue.get_nowait()

            # Pass through all events
            yield event

        # Yield any remaining queued filler events
        while not filler_queue.empty():
            yield filler_queue.get_nowait()

    finally:
        cancel_filler_timer()
