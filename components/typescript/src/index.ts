import dotenv from "dotenv";
import { fileURLToPath } from "node:url";
import { existsSync } from "node:fs";
import { createAgent, AIMessage, ToolMessage } from "langchain";
import path from "node:path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load .env from project root (two levels up from src/)
dotenv.config({ path: path.resolve(__dirname, "../../../.env") });
import { Hono } from "hono";
import { serve } from "@hono/node-server";
import { serveStatic } from "@hono/node-server/serve-static";
import { cors } from "hono/cors";
import { createNodeWebSocket } from "@hono/node-ws";
import type { WSContext } from "hono/ws";
import type WebSocket from "ws";
import { iife, writableIterator } from "./utils";
import { MemorySaver, interrupt, Command } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { v4 as uuidv4 } from "uuid";
import { CARTESIA_TTS_SYSTEM_PROMPT, CartesiaTTS } from "./cartesia";
import { AssemblyAISTT } from "./assemblyai/index";
import type { VoiceAgentEvent } from "./types";
import { thinkingFillerStream } from "./thinking-filler";

const STATIC_DIR = path.join(__dirname, "../../web/dist");
const PORT = parseInt(process.env.PORT ?? "8000");

if (!existsSync(STATIC_DIR)) {
  console.error(
    `Web build not found at ${STATIC_DIR}.\n` +
      "Run 'make build-web' or 'make dev-ts' from the project root."
  );
  process.exit(1);
}

const app = new Hono();
const { injectWebSocket, upgradeWebSocket } = createNodeWebSocket({ app });

app.use("/*", cors());

const addToOrder = tool(
  async ({ item, quantity }) => {
    // Demonstrate HITL: if user orders ham, interrupt and ask for alternative
    let finalItem = item;
    if (item.toLowerCase() === "ham") {
      finalItem = interrupt(
        "Sorry, we're out of ham today. Would you like turkey or roast beef instead?"
      );
      if (!["turkey", "roast beef"].includes(finalItem.toLowerCase())) {
        throw new Error(
          "Sorry, please choose either turkey or roast beef as the alternative."
        );
      }
    }
    return `Added ${quantity} x ${finalItem} to the order.`;
  },
  {
    name: "add_to_order",
    description: "Add an item to the customer's sandwich order.",
    schema: z.object({
      item: z.string(),
      quantity: z.number(),
    }),
  }
);

const confirmOrder = tool(
  async ({ orderSummary }) => {
    return `Order confirmed: ${orderSummary}. Sending to kitchen.`;
  },
  {
    name: "confirm_order",
    description: "Confirm the final order with the customer.",
    schema: z.object({
      orderSummary: z.string().describe("Summary of the order"),
    }),
  }
);

const hangUp = tool(
  ({ reason }) => {
    return `Call ended: ${reason}`;
  },
  {
    name: "hang_up",
    description:
      "End the call and hang up the connection. Use this when the conversation has naturally concluded, the customer says goodbye, or explicitly asks to end the call.",
    schema: z.object({
      reason: z
        .string()
        .describe(
          "Brief reason for ending the call (e.g., 'Order complete', 'Customer said goodbye')."
        ),
    }),
  }
);

const systemPrompt = `
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
`;

const agent = createAgent({
  model: "claude-haiku-4-5",
  tools: [addToOrder, confirmOrder, hangUp],
  checkpointer: new MemorySaver(),
  systemPrompt: systemPrompt,
});

interface STTStreamOptions {
  /**
   * Callback when speech is detected (for barge-in support).
   */
  onSpeechStart?: () => void;
}

/**
 * Transform stream: Audio (Uint8Array) → Voice Events (VoiceAgentEvent)
 *
 * This function takes a stream of audio chunks and sends them to AssemblyAI for STT.
 *
 * It uses a producer-consumer pattern where:
 * - Producer: Reads audio chunks from audioStream and sends them to AssemblyAI
 * - Consumer: Receives transcription events from AssemblyAI and yields them
 *
 * @param audioStream - Async iterator of PCM audio bytes (16-bit, mono, 16kHz)
 * @param options - Optional configuration including barge-in callback
 * @returns Async generator yielding STT events (stt_chunk for partials, stt_output for final transcripts)
 */
async function* sttStream(
  audioStream: AsyncIterable<Uint8Array>,
  options: STTStreamOptions = {}
): AsyncGenerator<VoiceAgentEvent> {
  const stt = new AssemblyAISTT({
    sampleRate: 16000,
    onSpeechStart: options.onSpeechStart,
  });
  const passthrough = writableIterator<VoiceAgentEvent>();

  /**
   * Promise that pumps audio chunks to AssemblyAI.
   *
   * This runs concurrently with the consumer, continuously reading audio
   * chunks from the input stream and forwarding them to AssemblyAI.
   * This allows transcription to begin before all audio has arrived.
   */
  const producer = iife(async () => {
    try {
      // Stream each audio chunk to AssemblyAI as it arrives
      for await (const audioChunk of audioStream) {
        await stt.sendAudio(audioChunk);
      }
    } finally {
      // Signal to AssemblyAI that audio streaming is complete
      await stt.close();
    }
  });

  /**
   * Promise that receives transcription events from AssemblyAI.
   *
   * This runs concurrently with the producer, listening for STT events
   * and pushing them into the passthrough iterator for downstream stages.
   */
  const consumer = iife(async () => {
    for await (const event of stt.receiveEvents()) {
      passthrough.push(event);
    }
  });

  try {
    // Yield events as they arrive from the consumer
    yield* passthrough;
  } finally {
    // Wait for the producer and consumer to complete when cleaning up
    await Promise.all([producer, consumer]);
  }
}

/**
 * Transform stream: Voice Events → Voice Events (with Agent Responses)
 *
 * This function takes a stream of upstream voice agent events and processes them.
 * When an stt_output event arrives, it passes the transcript to the LangChain agent.
 * The agent streams back its response tokens as agent_chunk events.
 * Tool calls and results are also emitted as separate events.
 * All other upstream events are passed through unchanged.
 *
 * Supports Human-In-The-Loop (HITL) interrupts. When the agent calls interrupt(),
 * the interrupt message is emitted as an agent_chunk and the next user input
 * will resume the graph with the user's response.
 *
 * @param eventStream - An async iterator of upstream voice agent events
 * @returns Async generator yielding all upstream events plus agent_chunk, tool_call, and tool_result events
 */
async function* agentStream(
  eventStream: AsyncIterable<VoiceAgentEvent>
): AsyncGenerator<VoiceAgentEvent> {
  // Generate a unique thread ID for this conversation session
  // This allows the agent to maintain conversation context across multiple turns
  // using the checkpointer (MemorySaver) configured in the agent
  const threadId = uuidv4();

  // Track if there's a pending interrupt (HITL)
  let pendingInterrupt: string | undefined;

  for await (const event of eventStream) {
    yield event;
    if (event.type === "stt_output") {
      console.log(`[AgentStream] Processing transcript: "${event.transcript}"`);

      // Determine input based on whether we're resuming from an interrupt
      let input: { messages: HumanMessage[] } | Command;

      if (pendingInterrupt !== undefined) {
        console.log(
          `[AgentStream] Resuming from interrupt with user response: "${event.transcript}"`
        );
        input = new Command({ resume: event.transcript });
        pendingInterrupt = undefined;
      } else {
        console.log(`[AgentStream] Sending new message to agent`);
        input = { messages: [new HumanMessage(event.transcript)] };
      }

      try {
        const stream = await agent.stream(input, {
          configurable: { thread_id: threadId },
          streamMode: "messages",
        });

        for await (const [message] of stream) {
          if (AIMessage.isInstance(message) && message.tool_calls) {
            yield { type: "agent_chunk", text: message.text, ts: Date.now() };
            for (const toolCall of message.tool_calls) {
              console.log(`[AgentStream] Tool call: ${toolCall.name}`);
              yield {
                type: "tool_call",
                id: toolCall.id ?? uuidv4(),
                name: toolCall.name,
                args: toolCall.args,
                ts: Date.now(),
              };
            }
          }
          if (ToolMessage.isInstance(message)) {
            const toolName = message.name ?? "unknown";
            const result =
              typeof message.content === "string"
                ? message.content
                : JSON.stringify(message.content);

            console.log(`[AgentStream] Tool result: ${toolName} -> ${result.substring(0, 100)}`);
            yield {
              type: "tool_result",
              toolCallId: message.tool_call_id ?? "",
              name: toolName,
              result,
              ts: Date.now(),
            };

            // Check if this is the hang_up tool - emit hang_up event
            if (toolName === "hang_up") {
              yield {
                type: "hang_up",
                reason: result,
                ts: Date.now(),
              };
            }
          }
        }

        console.log(`[AgentStream] Agent stream completed, checking for interrupts...`);

        // Check for interrupts after streaming completes (HITL support)
        interface StateTask {
          interrupts?: Array<{ value: unknown }>;
        }
        interface GraphState {
          tasks?: StateTask[];
        }

        const state = (await agent.getState({
          configurable: { thread_id: threadId },
        })) as GraphState;

        console.log(`[AgentStream] State tasks: ${state.tasks?.length ?? 0}`);

        if (state.tasks && state.tasks.length > 0) {
          for (const task of state.tasks) {
            if (task.interrupts && task.interrupts.length > 0) {
              const interruptValue = task.interrupts[0].value;
              const interruptMessage =
                typeof interruptValue === "string"
                  ? interruptValue
                  : String(interruptValue);

              console.log(`[AgentStream] Interrupt detected: "${interruptMessage}"`);
              pendingInterrupt = interruptMessage;

              // Emit the interrupt message as an agent_chunk so it goes through TTS
              yield {
                type: "agent_chunk",
                text: interruptMessage,
                ts: Date.now(),
              };

              // Also emit an interrupt event for tracking
              yield {
                type: "interrupt",
                message: interruptMessage,
                ts: Date.now(),
              };
            }
          }
        }

        // Signal that the agent has finished responding for this turn
        console.log(`[AgentStream] Emitting agent_end`);
        yield { type: "agent_end", ts: Date.now() };
      } catch (error) {
        console.error(`[AgentStream] Error processing agent:`, error);
        // Emit an error response so the user knows something went wrong
        yield {
          type: "agent_chunk",
          text: "I'm sorry, I encountered an error. Could you please repeat that?",
          ts: Date.now(),
        };
        yield { type: "agent_end", ts: Date.now() };
      }
    }
  }
}

interface TTSStreamOptions {
  /**
   * Optional TTS instance to use. If not provided, one will be created.
   * Providing an external instance allows for barge-in interruption.
   */
  tts?: CartesiaTTS;
}

/**
 * Transform stream: Voice Events → Voice Events (with Audio)
 *
 * This function takes a stream of upstream voice agent events and processes them.
 * When agent_chunk events arrive, it sends the text to Cartesia for TTS synthesis.
 * Audio is streamed back as tts_chunk events as it's generated.
 * All upstream events are passed through unchanged.
 *
 * It uses a producer-consumer pattern where:
 * - Producer: Reads events from eventStream, passes them through, and sends agent text to Cartesia
 * - Consumer: Receives audio chunks from Cartesia and yields them as tts_chunk events
 *
 * @param eventStream - An async iterator of upstream voice agent events
 * @param options - Optional configuration including external TTS instance for barge-in
 * @returns Async generator yielding all upstream events plus tts_chunk events for synthesized audio
 */
async function* ttsStream(
  eventStream: AsyncIterable<VoiceAgentEvent>,
  options: TTSStreamOptions = {}
): AsyncGenerator<VoiceAgentEvent> {
  const tts = options.tts ?? new CartesiaTTS({
    voiceId: "f6ff7c0c-e396-40a9-a70b-f7607edb6937",
  });
  const passthrough = writableIterator<VoiceAgentEvent>();

  /**
   * Promise that reads events from the upstream stream and sends text to Cartesia.
   *
   * This runs concurrently with the consumer, continuously reading events
   * from the upstream stream and forwarding agent text to Cartesia for synthesis.
   * All events are passed through to the downstream via the passthrough iterator.
   * This allows audio generation to begin before the agent has finished generating.
   */
  const producer = iife(async () => {
    try {
      let buffer: string[] = [];
      for await (const event of eventStream) {
        // Pass through all events to downstream consumers
        passthrough.push(event);
        // Send agent text chunks to Cartesia for synthesis
        if (event.type === "agent_chunk") {
          buffer.push(event.text);
        }
        // Send all buffered text to Cartesia for synthesis
        if (event.type === "agent_end") {
          await tts.sendText(buffer.join(""));
          buffer = [];
        }
      }
    } finally {
      // Signal to Cartesia that text sending is complete
      await tts.close();
    }
  });

  /**
   * Promise that receives audio events from Cartesia.
   *
   * This runs concurrently with the producer, listening for TTS audio chunks
   * and pushing them into the passthrough iterator for downstream stages.
   */
  const consumer = iife(async () => {
    for await (const event of tts.receiveEvents()) {
      passthrough.push(event);
    }
  });

  try {
    // Yield events as they arrive from both producer (upstream) and consumer (TTS)
    yield* passthrough;
  } finally {
    // Wait for the producer and consumer to complete when cleaning up
    await Promise.all([producer, consumer]);
  }
}

app.get("/*", serveStatic({ root: STATIC_DIR }));

app.get(
  "/ws",
  upgradeWebSocket(async () => {
    let currentSocket: WSContext<WebSocket> | undefined;

    // Create a writable stream for incoming WebSocket audio data
    const inputStream = writableIterator<Uint8Array>();

    // Create TTS instance for barge-in support (can be interrupted)
    const tts = new CartesiaTTS({
      voiceId: "f6ff7c0c-e396-40a9-a70b-f7607edb6937",
      onInterrupt: () => {
        console.log("[Pipeline] TTS interrupted (barge-in)");
        // Notify client to clear audio buffer
        currentSocket?.send(JSON.stringify({ type: "clear_audio", ts: Date.now() }));
      },
    });

    // Define the voice processing pipeline as a chain of async generators
    // Audio -> STT events (with barge-in callback to interrupt TTS)
    const transcriptEventStream = sttStream(inputStream, {
      onSpeechStart: () => {
        console.log("[Pipeline] Speech detected, interrupting TTS (barge-in)");
        tts.interrupt();
      },
    });
    // STT events -> STT Events + Agent events
    const agentEventStream = agentStream(transcriptEventStream);
    // Agent events -> Agent events with filler phrases when agent takes too long
    const fillerEventStream = thinkingFillerStream(agentEventStream, {
      thresholdMs: 1200,
      onFillerEmitted: (phrase) => {
        console.log(`[Pipeline] Filler emitted: "${phrase}"`);
      },
    });
    // STT events + Agent events -> STT Events + Agent Events + TTS events
    const outputEventStream = ttsStream(fillerEventStream, { tts });

    // Track if a hang_up event was received
    let pendingHangUp = false;

    const flushPromise = iife(async () => {
      // Process all events from the pipeline, sending events back to the client
      for await (const event of outputEventStream) {
        currentSocket?.send(JSON.stringify(event));

        // Check for hang_up event - close connection after TTS completes
        if (event.type === "hang_up") {
          console.log(`[WebSocket] Hang up requested: ${event.reason}`);
          pendingHangUp = true;
        }

        // Close the connection after TTS finishes synthesizing the final audio
        if (pendingHangUp && event.type === "tts_end") {
          // Give client a brief moment to receive the final audio chunk
          setTimeout(() => {
            console.log("[WebSocket] Closing connection after final TTS");
            inputStream.cancel();
            currentSocket?.close(1000, "Call ended by agent");
          }, 500);
        }
      }
    });

    return {
      onOpen(_, ws) {
        currentSocket = ws;
      },
      onMessage(event) {
        // Push incoming audio data into the pipeline's input stream
        const data = event.data;
        if (Buffer.isBuffer(data)) {
          inputStream.push(new Uint8Array(data));
        } else if (data instanceof ArrayBuffer) {
          inputStream.push(new Uint8Array(data));
        }
      },
      async onClose() {
        // Signal end of stream when socket closes
        inputStream.cancel();
        await flushPromise;
      },
    };
  })
);

const server = serve({
  fetch: app.fetch,
  port: PORT,
});

injectWebSocket(server);

console.log(`Server is running on port ${PORT}`);
