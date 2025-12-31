import WebSocket from "ws";
import { writableIterator } from "../utils";
import type {
  CartesiaTTSRequest,
  CartesiaTTSResponse,
  CartesiaOutputFormat,
  CartesiaVoice,
} from "./api-types";
import type { VoiceAgentEvent } from "../types";

type TTSEvent = VoiceAgentEvent.TTSChunk | VoiceAgentEvent.TTSEnd;

interface CartesiaTTSOptions {
  apiKey?: string;
  voiceId?: string;
  modelId?: string;
  sampleRate?: number;
  encoding?: CartesiaOutputFormat["encoding"];
  language?: string;
  cartesiaVersion?: string;
  /**
   * Callback called when TTS output is interrupted (for barge-in).
   */
  onInterrupt?: () => void;
}

export class CartesiaTTS {
  apiKey: string;
  voiceId: string;
  modelId: string;
  sampleRate: number;
  encoding: CartesiaOutputFormat["encoding"];
  language: string;
  cartesiaVersion: string;
  onInterrupt?: () => void;

  protected _bufferIterator = writableIterator<TTSEvent>();
  protected _connectionPromise: Promise<WebSocket> | null = null;
  protected _contextCounter = 0;
  protected _isInterrupted = false;

  /**
   * Generate a valid context_id for Cartesia.
   * Context IDs must only contain alphanumeric characters, underscores, and hyphens.
   */
  protected _generateContextId(): string {
    const timestamp = Date.now();
    const counter = this._contextCounter++;
    return `ctx_${timestamp}_${counter}`;
  }

  protected get _connection(): Promise<WebSocket> {
    if (this._connectionPromise) {
      return this._connectionPromise;
    }

    this._connectionPromise = new Promise((resolve, reject) => {
      const params = new URLSearchParams({
        api_key: this.apiKey,
        cartesia_version: this.cartesiaVersion,
      });
      const url = `wss://api.cartesia.ai/tts/websocket?${params.toString()}`;
      const ws = new WebSocket(url);

      ws.on("open", () => {
        resolve(ws);
      });

      ws.on("message", (data: WebSocket.RawData) => {
        try {
          const message: CartesiaTTSResponse = JSON.parse(data.toString());

          if (message.data) {
            this._bufferIterator.push({
              type: "tts_chunk",
              audio: message.data,
              ts: Date.now(),
            });
          }

          // Emit tts_end when Cartesia signals synthesis is complete
          if (message.done || message.type === "done") {
            this._bufferIterator.push({
              type: "tts_end",
              ts: Date.now(),
            });
          }

          if (message.error) {
            throw new Error(`Cartesia error: ${message.error}`);
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

      ws.on("close", () => {
        this._connectionPromise = null;
      });
    });

    return this._connectionPromise;
  }

  constructor(options: CartesiaTTSOptions = {}) {
    this.apiKey = options.apiKey ?? process.env.CARTESIA_API_KEY ?? "";
    if (!this.apiKey) {
      throw new Error("Cartesia API key is required");
    }
    this.voiceId = options.voiceId ?? "f6ff7c0c-e396-40a9-a70b-f7607edb6937";
    this.modelId = options.modelId ?? "sonic-3";
    this.sampleRate = options.sampleRate ?? 24000;
    this.encoding = options.encoding ?? "pcm_s16le";
    this.language = options.language ?? "en";
    this.cartesiaVersion = options.cartesiaVersion ?? "2025-04-16";
    this.onInterrupt = options.onInterrupt;
  }

  /**
   * Interrupt the current TTS output (for barge-in support).
   * Closes the connection to stop audio generation and clears any pending audio.
   */
  async interrupt(): Promise<void> {
    if (this._isInterrupted) return;

    console.log("[CartesiaTTS] Interrupted by user (barge-in)");
    this._isInterrupted = true;

    // Close connection to stop audio generation
    if (this._connectionPromise) {
      try {
        const ws = await this._connectionPromise;
        ws.close();
      } catch {
        // Ignore close errors
      }
      this._connectionPromise = null;
    }

    // Call the onInterrupt callback
    this.onInterrupt?.();

    // Reset interrupted state after a brief delay to allow new input
    setTimeout(() => {
      this._isInterrupted = false;
    }, 100);
  }

  async sendText(text: string): Promise<void> {
    if (!text || !text.trim()) {
      return;
    }

    const conn = await this._connection;
    if (conn.readyState === WebSocket.OPEN) {
      const voice: CartesiaVoice = {
        mode: "id",
        id: this.voiceId,
      };

      const outputFormat: CartesiaOutputFormat = {
        container: "raw",
        encoding: this.encoding,
        sample_rate: this.sampleRate,
      };

      const payload: CartesiaTTSRequest = {
        model_id: this.modelId,
        transcript: text,
        voice: voice,
        output_format: outputFormat,
        language: this.language,
        context_id: this._generateContextId(),
      };
      conn.send(JSON.stringify(payload));
    }
  }

  async *receiveEvents(): AsyncGenerator<TTSEvent> {
    yield* this._bufferIterator;
  }

  async close(): Promise<void> {
    if (this._connectionPromise) {
      const ws = await this._connectionPromise;
      ws.close();
    }
  }
}
