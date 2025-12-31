import { writableIterator, iife } from "./utils";
import type { VoiceAgentEvent } from "./types";

/**
 * Options for the thinking filler transform.
 */
export interface ThinkingFillerOptions {
  /**
   * Time in milliseconds before emitting a filler phrase.
   * @default 1200
   */
  thresholdMs?: number;

  /**
   * Array of filler phrases to randomly choose from.
   */
  fillerPhrases?: string[];

  /**
   * Whether the filler functionality is enabled.
   * @default true
   */
  enabled?: boolean;

  /**
   * Callback when a filler phrase is emitted.
   */
  onFillerEmitted?: (phrase: string) => void;
}

const DEFAULT_FILLER_PHRASES = [
  "Let me see here...",
  "Hmm, one moment...",
  "Ah, let me think...",
  "Just a second...",
  "Mhm, okay...",
  "Let me check that...",
];

/**
 * Transform stream: Voice Events → Voice Events (with Filler Phrases)
 *
 * This function adds "thinking filler" phrases when the agent takes longer
 * than a threshold to respond. This creates a more natural, conversational
 * experience for voice applications.
 *
 * When an stt_output event arrives, it starts a timer. If no agent_chunk
 * event arrives within the threshold, it emits a filler phrase as an
 * agent_chunk event. The timer is cancelled when agent_chunk arrives.
 *
 * @param eventStream - An async iterator of upstream voice agent events
 * @param options - Configuration options for the filler behavior
 * @returns Async generator yielding all upstream events plus potential filler phrases
 */
export async function* thinkingFillerStream(
  eventStream: AsyncIterable<VoiceAgentEvent>,
  options: ThinkingFillerOptions = {}
): AsyncGenerator<VoiceAgentEvent> {
  const {
    thresholdMs = 1200,
    fillerPhrases = DEFAULT_FILLER_PHRASES,
    enabled = true,
    onFillerEmitted,
  } = options;

  if (!enabled) {
    yield* eventStream;
    return;
  }

  // State for tracking filler timer
  let fillerTimeout: ReturnType<typeof setTimeout> | null = null;
  let fillerEmittedThisTurn = false;
  let waitingForAgentResponse = false;

  // Create a passthrough iterator to allow injecting filler phrases
  const passthrough = writableIterator<VoiceAgentEvent>();

  // Helper to pick a random filler phrase
  const pickFillerPhrase = () =>
    fillerPhrases[Math.floor(Math.random() * fillerPhrases.length)];

  // Helper to clear the filler timer
  const clearFillerTimer = () => {
    if (fillerTimeout) {
      clearTimeout(fillerTimeout);
      fillerTimeout = null;
    }
  };

  // Helper to emit a filler phrase
  const emitFiller = () => {
    if (fillerEmittedThisTurn || !waitingForAgentResponse) return;

    const phrase = pickFillerPhrase();
    console.log(`[ThinkingFiller] Emitting filler: "${phrase}"`);

    passthrough.push({
      type: "agent_chunk",
      text: phrase,
      ts: Date.now(),
    });

    // Emit agent_end to trigger TTS for the filler
    passthrough.push({
      type: "agent_end",
      ts: Date.now(),
    });

    fillerEmittedThisTurn = true;
    onFillerEmitted?.(phrase);
  };

  // Producer: process upstream events and manage filler timer
  const producer = iife(async () => {
    try {
      for await (const event of eventStream) {
        // When we receive stt_output, start the filler timer
        if (event.type === "stt_output") {
          fillerEmittedThisTurn = false;
          waitingForAgentResponse = true;
          clearFillerTimer();

          console.log(
            `[ThinkingFiller] STT output received, starting ${thresholdMs}ms timer`
          );

          fillerTimeout = setTimeout(() => {
            emitFiller();
          }, thresholdMs);
        }

        // When we receive agent_chunk, cancel the filler timer
        if (event.type === "agent_chunk") {
          waitingForAgentResponse = false;
          clearFillerTimer();
        }

        // Pass through all events
        passthrough.push(event);
      }
    } finally {
      clearFillerTimer();
      passthrough.cancel();
    }
  });

  try {
    yield* passthrough;
  } finally {
    clearFillerTimer();
    await producer;
  }
}
