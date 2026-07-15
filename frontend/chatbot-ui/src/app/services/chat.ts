import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';


// Injectable means this service can be injected into components
@Injectable({
  providedIn: 'root', // This makes the service available application-wide without needing to add it to a module's providers array
})
export class Chat {
  private apiUrl = 'http://localhost:8000';  // Base URL for the backend API

  constructor(private http: HttpClient) {}

  // Fetches prior chat history for a session (used to rehydrate the panel
  // on a returning visit, when Session already has a persisted session_id).
  getConversation(sessionId: string): Observable<{ session_id: string; history: { role: string; content: string }[]; message_count: number }> {
    return this.http.get<any>(`${this.apiUrl}/conversation/${sessionId}`);
  }

  send(userPrompt: string, sessionId: string): Observable<ChatResponse> {
    return this.http
      .post<any>(`${this.apiUrl}/chat`, {
        session_id: sessionId,
        message: userPrompt,
      })
      .pipe(
        map((response) => ({ reply: response.response }))
      );
  }

  // Streams the assistant reply token by token via Server-Sent Events.
  // Calls onToken for each text chunk and onDone when the stream ends.
  async sendStream(
    userPrompt: string,
    sessionId: string,
    onTraceId: (traceId: string) => void,
    onToken: (token: string) => void,
    onDone: (fullText: string) => void,
    onError: (err: string) => void,
  ): Promise<void> {
    const response = await fetch(`${this.apiUrl}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, message: userPrompt }),
    });

    if (!response.ok || !response.body) {
      onError(`Request failed: ${response.status}`);
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullText = '';
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (payload === '[DONE]') {
          onDone(fullText);
          return;
        }
        try {
          const parsed = JSON.parse(payload);
          if (parsed.error) {
            onError(parsed.error);
            return;
          }
          if (parsed.trace_id) {
            onTraceId(parsed.trace_id);
            continue;
          }
          if (parsed.text) {
            fullText += parsed.text;
            onToken(parsed.text);
          }
        } catch {
          // malformed chunk — skip
        }
      }
    }

    onDone(fullText);
  }

  sendFeedback(traceId: string, value: boolean, comment?: string) {
    return this.http.post(`${this.apiUrl}/feedback`, { trace_id: traceId, value, comment });
  }
}

export interface ChatResponse {
  reply: string;
}
