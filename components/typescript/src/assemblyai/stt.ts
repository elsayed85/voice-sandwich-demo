import WebSocket from "ws";
import { writableIterator } from "../utils";
import type { AssemblyAISTTMessage } from "./api-types";
import type { VoiceAgentEvent } from "../types";

/**
 * Endpointing configuration for turn detection.
 * Controls how the STT detects when the user has finished speaking.
 */
export interface EndpointingConfig {
  /**
   * Confidence threshold for semantic end-of-turn detection (0-1).
   * Lower = more aggressive (faster responses), Higher = more conservative.
   * @default 0.4
   */
  endOfTurnConfidenceThreshold?: number;

  /**
   * Minimum silence duration (ms) when confident about end of turn.
   * Lower = faster response when STT is confident the user finished.
   * @default 400
   */
  minEndOfTurnSilenceWhenConfident?: number;

  /**
   * Maximum silence duration (ms) before forcing end-of-turn.
   * Acts as a fallback when semantic detection isn't confident.
   * @default 1280
   */
  maxTurnSilence?: number;
}

interface AssemblyAISTTOptions {
  apiKey?: string;
  sampleRate?: number;
  formatTurns?: boolean;
  /**
   * Endpointing configuration for turn detection.
   * Controls how aggressively the STT detects end of speech.
   */
  endpointing?: EndpointingConfig;
  /**
   * Callback when speech is detected (partial transcript received).
   * Useful for implementing barge-in to interrupt TTS.
   */
  onSpeechStart?: () => void;
}

export class AssemblyAISTT {
  apiKey: string;
  sampleRate: number;
  formatTurns: boolean;
  endpointing: EndpointingConfig;
  onSpeechStart?: () => void;

  protected _bufferIterator = writableIterator<VoiceAgentEvent.STTEvent>();
  protected _connectionPromise: Promise<WebSocket> | null = null;
  // Track if we've already signaled speech start for current utterance
  protected _speechStartSignaled = false;

  protected get _connection(): Promise<WebSocket> {
    if (this._connectionPromise) {
      return this._connectionPromise;
    }

    this._connectionPromise = new Promise((resolve, reject) => {
      const params = new URLSearchParams({
        sample_rate: this.sampleRate.toString(),
        format_turns: this.formatTurns.toString().toLowerCase(),
      });

      // Add endpointing configuration if provided
      if (this.endpointing.endOfTurnConfidenceThreshold !== undefined) {
        params.set("end_of_turn_confidence_threshold", this.endpointing.endOfTurnConfidenceThreshold.toString());
      }
      if (this.endpointing.minEndOfTurnSilenceWhenConfident !== undefined) {
        params.set("min_end_of_turn_silence_when_confident", this.endpointing.minEndOfTurnSilenceWhenConfident.toString());
      }
      if (this.endpointing.maxTurnSilence !== undefined) {
        params.set("max_turn_silence", this.endpointing.maxTurnSilence.toString());
      }

      const url = `wss://streaming.assemblyai.com/v3/ws?${params.toString()}`;
      const ws = new WebSocket(url, {
        headers: { Authorization: this.apiKey },
      });

      ws.on("open", () => {
        resolve(ws);
      });

      ws.on("message", (data: WebSocket.RawData) => {
        try {
          const message: AssemblyAISTTMessage = JSON.parse(data.toString());
          if (message.type === "Begin") {
            // no-op
          } else if (message.type === "Turn") {
            if (message.turn_is_formatted) {
              if (message.transcript) {
                console.log(`[AssemblyAI STT] Final transcript: "${message.transcript}"`);
                this._bufferIterator.push({ type: "stt_output", transcript: message.transcript, ts: Date.now() });
              }
              // Reset speech start flag for next utterance
              this._speechStartSignaled = false;
            } else {
              this._bufferIterator.push({ type: "stt_chunk", transcript: message.transcript, ts: Date.now() });
              // Signal speech start for barge-in (only once per utterance)
              if (!this._speechStartSignaled && message.transcript?.trim().length > 0) {
                this._speechStartSignaled = true;
                this.onSpeechStart?.();
              }
            }
          } else if (message.type === "Termination") {
            console.log("[AssemblyAI STT] Received termination message");
          } else if (message.type === "Error") {
            throw new Error(message.error);
          }
        } catch (error) {
          // TODO: better catch json parsing error
          console.error(error);
        }
      });

      ws.on("error", (error) => {
        this._bufferIterator.cancel();
        reject(error);
      });

      ws.on("close", (code, reason) => {
        console.log(`[AssemblyAI STT] WebSocket closed: code=${code}, reason=${reason?.toString() || "none"}`);
        this._connectionPromise = null;
      });
    });

    return this._connectionPromise;
  }

  constructor(options: AssemblyAISTTOptions) {
    this.apiKey = options.apiKey || process.env.ASSEMBLYAI_API_KEY || "";
    this.sampleRate = options.sampleRate || 16000;
    this.formatTurns = options.formatTurns ?? true;
    this.endpointing = options.endpointing || {};
    this.onSpeechStart = options.onSpeechStart;

    if (!this.apiKey) {
      throw new Error("AssemblyAI API key is required");
    }
  }

  async sendAudio(buffer: Uint8Array): Promise<void> {
    try {
      const conn = await this._connection;
      if (conn.readyState === WebSocket.OPEN) {
        conn.send(buffer);
      } else {
        console.warn(`[AssemblyAI STT] Cannot send audio, WebSocket state: ${conn.readyState}`);
      }
    } catch (error) {
      console.error("[AssemblyAI STT] Error sending audio:", error);
    }
  }

  async *receiveEvents(): AsyncGenerator<VoiceAgentEvent.STTEvent> {
    yield* this._bufferIterator;
  }

  async close(): Promise<void> {
    if (this._connectionPromise) {
      const ws = await this._connectionPromise;
      ws.close();
    }
  }
}
