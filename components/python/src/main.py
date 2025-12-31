import asyncio
import contextlib
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableGenerator
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from starlette.staticfiles import StaticFiles

from assemblyai_stt import AssemblyAISTT
from cartesia_tts import CartesiaTTS
from events import (
    AgentChunkEvent,
    AgentEndEvent,
    ClearAudioEvent,
    HangUpEvent,
    InterruptEvent,
    ToolCallEvent,
    ToolResultEvent,
    VoiceAgentEvent,
    event_to_dict,
)
from thinking_filler import thinking_filler_stream
from utils import merge_async_iters

load_dotenv()

# Static files are served from the shared web build output
STATIC_DIR = Path(__file__).parent.parent.parent / "web" / "dist"

if not STATIC_DIR.exists():
    raise RuntimeError(
        f"Web build not found at {STATIC_DIR}. "
        "Run 'make build-web' or 'make dev-py' from the project root."
    )

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def add_to_order(item: str, quantity: int) -> str:
    """Add an item to the customer's sandwich order."""
    # Demonstrate HITL: if user orders ham, interrupt and ask for alternative
    final_item = item
    if item.lower() == "ham":
        final_item = interrupt(
            "Sorry, we're out of ham today. Would you like turkey or roast beef instead?"
        )
        if final_item.lower() not in ["turkey", "roast beef"]:
            raise ValueError(
                "Sorry, please choose either turkey or roast beef as the alternative."
            )
    return f"Added {quantity} x {final_item} to the order."


def confirm_order(order_summary: str) -> str:
    """Confirm the final order with the customer."""
    return f"Order confirmed: {order_summary}. Sending to kitchen."


def hang_up(reason: str) -> str:
    """End the call and hang up the connection.

    Use this when the conversation has naturally concluded, the customer says goodbye,
    or explicitly asks to end the call.

    Args:
        reason: Brief reason for ending the call (e.g., 'Order complete', 'Customer said goodbye').
    """
    return f"Call ended: {reason}"


system_prompt = """
You are a helpful sandwich shop assistant. Your goal is to take the user's order.
Be concise and friendly.

Available toppings: lettuce, tomato, onion, pickles, mayo, mustard.
Available meats: turkey, ham, roast beef.
Available cheeses: swiss, cheddar, provolone.

IMPORTANT: You MUST call the hang_up tool in these situations:
- After confirming an order and the customer indicates they're done (says "no" to additional items, says goodbye, etc.)
- When the customer explicitly says goodbye, thanks you, or indicates the conversation is over
- When the customer says phrases like "that's it", "that's all", "I'm good", "bye", "thanks", "thank you"

Always call hang_up AFTER giving your final farewell message. Do not just respond with text - you must use the tool to properly end the call.

${CARTESIA_TTS_SYSTEM_PROMPT}
"""

agent = create_agent(
    model="anthropic:claude-haiku-4-5",
    tools=[add_to_order, confirm_order, hang_up],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
)


def create_stt_stream(
    on_speech_start: callable = None,
):
    """
    Factory function to create an STT stream with optional barge-in support.

    Args:
        on_speech_start: Optional callback when speech is detected (for barge-in).
                         Should return an async generator that yields ClearAudioEvent.
    """

    async def _stt_stream(
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[VoiceAgentEvent]:
        """
        Transform stream: Audio (Bytes) → Voice Events (VoiceAgentEvent)

        This function takes a stream of audio chunks and sends them to AssemblyAI for STT.

        It uses a producer-consumer pattern where:
        - Producer: A background task reads audio chunks from audio_stream and sends
          them to AssemblyAI via WebSocket. This runs concurrently with the consumer,
          allowing transcription to begin before all audio has arrived.
        - Consumer: The main coroutine receives transcription events from AssemblyAI
          and yields them downstream. Events include both partial results (stt_chunk)
          and final transcripts (stt_output).

        Args:
            audio_stream: Async iterator of PCM audio bytes (16-bit, mono, 16kHz)

        Yields:
            STT events (stt_chunk for partials, stt_output for final transcripts)
        """
        # Queue to emit clear_audio events when barge-in is detected
        clear_audio_queue: asyncio.Queue[ClearAudioEvent] = asyncio.Queue()

        def handle_speech_start():
            """Called when speech is detected - triggers barge-in."""
            if on_speech_start:
                on_speech_start()
            # Queue a clear_audio event to be emitted
            clear_audio_queue.put_nowait(ClearAudioEvent.create())

        stt = AssemblyAISTT(sample_rate=16000, on_speech_start=handle_speech_start)

        async def send_audio():
            """
            Background task that pumps audio chunks to AssemblyAI.

            This runs concurrently with the main coroutine, continuously reading
            audio chunks from the input stream and forwarding them to AssemblyAI.
            When the input stream ends, it signals completion by closing the
            WebSocket connection.
            """
            try:
                # Stream each audio chunk to AssemblyAI as it arrives
                async for audio_chunk in audio_stream:
                    await stt.send_audio(audio_chunk)
            finally:
                # Signal to AssemblyAI that audio streaming is complete
                await stt.close()

        # Launch the audio sending task in the background
        # This allows us to simultaneously receive transcripts in the main coroutine
        send_task = asyncio.create_task(send_audio())

        try:
            # Consumer loop: receive and yield transcription events as they arrive
            # from AssemblyAI. The receive_events() method listens on the WebSocket
            # for transcript events and yields them as they become available.
            async for event in stt.receive_events():
                # Check for any pending clear_audio events first
                while not clear_audio_queue.empty():
                    yield clear_audio_queue.get_nowait()
                yield event
        finally:
            # Cleanup: ensure the background task is cancelled and awaited
            with contextlib.suppress(asyncio.CancelledError):
                send_task.cancel()
                await send_task
            # Ensure the WebSocket connection is closed
            await stt.close()

    return _stt_stream


async def _agent_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Voice Events → Voice Events (with Agent Responses)

    This function takes a stream of upstream voice agent events and processes them.
    When an stt_output event arrives, it passes the transcript to the LangChain agent.
    The agent streams back its response tokens as agent_chunk events.
    Tool calls and results are also emitted as separate events.
    All other upstream events are passed through unchanged.

    Supports Human-In-The-Loop (HITL) interrupts. When the agent calls interrupt(),
    the interrupt message is emitted as an agent_chunk and the next user input
    will resume the graph with the user's response.

    Args:
        event_stream: An async iterator of upstream voice agent events

    Yields:
        All upstream events plus agent_chunk, tool_call, and tool_result events
    """
    # Generate a unique thread ID for this conversation session
    # This allows the agent to maintain conversation context across multiple turns
    # using the checkpointer (InMemorySaver) configured in the agent
    thread_id = str(uuid4())

    # Track if there's a pending interrupt (HITL)
    pending_interrupt: str | None = None

    # Process each event as it arrives from the upstream STT stage
    async for event in event_stream:
        # Pass through all events to downstream consumers
        yield event

        # When we receive a final transcript, invoke the agent
        if event.type == "stt_output":
            # Determine input based on whether we're resuming from an interrupt
            if pending_interrupt is not None:
                print(
                    f'[AgentStream] Resuming from interrupt with user response: "{event.transcript}"'
                )
                agent_input = Command(resume=event.transcript)
                pending_interrupt = None
            else:
                agent_input = {"messages": [HumanMessage(content=event.transcript)]}

            # Stream the agent's response using LangChain's astream method.
            # stream_mode="messages" yields message chunks as they're generated.
            stream = agent.astream(
                agent_input,
                {"configurable": {"thread_id": thread_id}},
                stream_mode="messages",
            )

            # Iterate through the agent's streaming response. The stream yields
            # tuples of (message, metadata), but we only need the message.
            async for message, _ in stream:
                # Emit agent chunks (AI messages)
                if isinstance(message, AIMessage):
                    # Extract and yield the text content from each message chunk
                    yield AgentChunkEvent.create(message.text)
                    # Emit tool calls if present
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            yield ToolCallEvent.create(
                                id=tool_call.get("id", str(uuid4())),
                                name=tool_call.get("name", "unknown"),
                                args=tool_call.get("args", {}),
                            )

                # Emit tool results (tool messages)
                if isinstance(message, ToolMessage):
                    tool_name = getattr(message, "name", "unknown")
                    result = str(message.content) if message.content else ""

                    yield ToolResultEvent.create(
                        tool_call_id=getattr(message, "tool_call_id", ""),
                        name=tool_name,
                        result=result,
                    )

                    # Check if this is the hang_up tool - emit hang_up event
                    if tool_name == "hang_up":
                        yield HangUpEvent.create(reason=result)

            # Check for interrupts after streaming completes (HITL support)
            state = await agent.aget_state({"configurable": {"thread_id": thread_id}})

            if hasattr(state, "tasks") and state.tasks:
                for task in state.tasks:
                    if hasattr(task, "interrupts") and task.interrupts:
                        interrupt_value = task.interrupts[0].value
                        interrupt_message = (
                            interrupt_value
                            if isinstance(interrupt_value, str)
                            else str(interrupt_value)
                        )

                        print(f'[AgentStream] Interrupt detected: "{interrupt_message}"')
                        pending_interrupt = interrupt_message

                        # Emit the interrupt message as an agent_chunk so it goes through TTS
                        yield AgentChunkEvent.create(interrupt_message)

                        # Also emit an interrupt event for tracking
                        yield InterruptEvent.create(message=interrupt_message)

            # Signal that the agent has finished responding for this turn
            yield AgentEndEvent.create()


def create_tts_stream(tts: CartesiaTTS):
    """
    Factory function to create a TTS stream with a specific TTS instance.

    Args:
        tts: CartesiaTTS instance to use for synthesis.
    """

    async def _tts_stream(
        event_stream: AsyncIterator[VoiceAgentEvent],
    ) -> AsyncIterator[VoiceAgentEvent]:
        """
        Transform stream: Voice Events → Voice Events (with Audio)

        This function takes a stream of upstream voice agent events and processes them.
        When agent_chunk events arrive, it sends the text to Cartesia for TTS synthesis.
        Audio is streamed back as tts_chunk events as it's generated.
        All upstream events are passed through unchanged.

        It uses merge_async_iters to combine two concurrent streams:
        - process_upstream(): Iterates through incoming events, yields them for
          passthrough, and sends agent text chunks to Cartesia for synthesis.
        - tts.receive_events(): Yields audio chunks from Cartesia as they are
          synthesized.

        The merge utility runs both iterators concurrently, yielding items from
        either stream as they become available. This allows audio generation to
        begin before the agent has finished generating all text, minimizing latency.

        Args:
            event_stream: An async iterator of upstream voice agent events

        Yields:
            All upstream events plus tts_chunk events for synthesized audio
        """
        async def process_upstream() -> AsyncIterator[VoiceAgentEvent]:
            """
            Process upstream events, yielding them while sending text to Cartesia.

            This async generator serves two purposes:
            1. Pass through all upstream events (stt_chunk, stt_output, agent_chunk)
               so downstream consumers can observe the full event stream.
            2. Buffer agent_chunk text and send to Cartesia when agent_end arrives.
               This ensures the full response is sent at once for better TTS quality.
            """
            buffer: list[str] = []
            async for event in event_stream:
                # Pass through all events to downstream consumers
                yield event
                # Buffer agent text chunks
                if event.type == "agent_chunk":
                    buffer.append(event.text)
                # Send all buffered text to Cartesia when agent finishes
                if event.type == "agent_end":
                    await tts.send_text("".join(buffer))
                    buffer = []

        try:
            # Merge the processed upstream events with TTS audio events
            # Both streams run concurrently, yielding events as they arrive
            async for event in merge_async_iters(process_upstream(), tts.receive_events()):
                yield event
        finally:
            # Cleanup: close the WebSocket connection to Cartesia
            await tts.close()

    return _tts_stream


async def _thinking_filler_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Voice Events → Voice Events (with Filler Phrases)

    Adds "thinking filler" phrases when the agent takes longer than a threshold
    to respond. This creates a more natural, conversational experience.
    """
    async for event in thinking_filler_stream(
        event_stream,
        threshold_ms=1200,
        on_filler_emitted=lambda phrase: print(f'[Pipeline] Filler emitted: "{phrase}"'),
    ):
        yield event


def create_pipeline(tts: CartesiaTTS):
    """
    Create a voice pipeline with barge-in support.

    Args:
        tts: CartesiaTTS instance to use for synthesis and barge-in interruption.
    """
    # Create STT stream with barge-in: when speech is detected, interrupt TTS
    def on_speech_start():
        print("[Pipeline] Speech detected - triggering barge-in")
        asyncio.create_task(tts.interrupt())

    return (
        RunnableGenerator(create_stt_stream(on_speech_start=on_speech_start))  # Audio -> STT events
        | RunnableGenerator(_agent_stream)  # STT events -> STT + Agent events
        | RunnableGenerator(_thinking_filler_stream)  # Add filler phrases when agent is slow
        | RunnableGenerator(create_tts_stream(tts))  # STT + Agent events -> All events
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Create TTS instance first so we can wire barge-in
    # The on_interrupt callback logs when TTS is interrupted
    tts = CartesiaTTS(
        on_interrupt=lambda: print("[Pipeline] TTS interrupted - clearing audio")
    )

    # Create the pipeline with barge-in support
    pipeline = create_pipeline(tts)

    async def websocket_audio_stream() -> AsyncIterator[bytes]:
        """Async generator that yields audio bytes from the websocket."""
        while True:
            data = await websocket.receive_bytes()
            yield data

    output_stream = pipeline.atransform(websocket_audio_stream())

    # Track if a hang_up event was received
    pending_hang_up = False

    # Process all events from the pipeline, sending events back to the client
    async for event in output_stream:
        await websocket.send_json(event_to_dict(event))

        # Check for hang_up event - close connection after TTS completes
        if hasattr(event, "type") and event.type == "hang_up":
            print(f"[WebSocket] Hang up requested: {event.reason}")
            pending_hang_up = True

        # Close the connection after TTS finishes synthesizing the final audio
        if pending_hang_up and hasattr(event, "type") and event.type == "tts_end":
            # Give client a brief moment to receive the final audio chunk
            await asyncio.sleep(0.5)
            print("[WebSocket] Closing connection after final TTS")
            await websocket.close(1000, "Call ended by agent")
            break


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
