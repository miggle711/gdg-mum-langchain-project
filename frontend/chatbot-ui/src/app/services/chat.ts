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
  private conversationId: string = ''; // Store the current conversation ID

  constructor(private http: HttpClient) {} 

  // an observable is a stream of data that components can subscribe (listen) to, allowing them to react to new data as it arrives without needing to poll for changes
  startConversation(): Observable<{ conversation_id: string; message: string }> {
    /**
     * This method starts a new conversation by making a POST request to the backend's /chat/start endpoint.
     * 
     * args: none
     * returns: an Observable that emits the conversation ID and initial message from the backend
     */
    console.log('Starting new conversation...');
    // We make a POST request to the backend to start a new conversation, 
    // which returns a conversation ID and an initial message
    return this.http.post<any>(`${this.apiUrl}/chat/start`, {}).pipe(
      map((response) => { // we destructure the response to get the conversation ID and initial message
        console.log('Conversation started with ID:', response.conversation_id);
        this.conversationId = response.conversation_id;  // we store the conversation ID in the service so it can be used for subsequent messages
        return response;
      })
    );
  }

  send(userPrompt: string): Observable<ChatResponse> {
    if (!this.conversationId) {
      throw new Error('Conversation not started. Call startConversation() first.');
    }
    return this.http
      .post<any>(`${this.apiUrl}/chat`, {
        conversation_id: this.conversationId,
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
    onTraceId: (traceId: string) => void,
    onToken: (token: string) => void,
    onDone: (fullText: string) => void,
    onError: (err: string) => void,
  ): Promise<void> {
    if (!this.conversationId) {
      throw new Error('Conversation not started. Call startConversation() first.');
    }

    const response = await fetch(`${this.apiUrl}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ conversation_id: this.conversationId, message: userPrompt }),
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

  // Getter and setter for the conversation ID, which can be useful if we want to manage the conversation ID separately or need to reset it when starting a new conversation
  getConversationId(): string {
    return this.conversationId;
  }

  setConversationId(id: string): void {
    this.conversationId = id;
  }

  sendFeedback(traceId: string, value: boolean, comment?: string) {
    return this.http.post(`${this.apiUrl}/feedback`, { trace_id: traceId, value, comment });
  }
}

export interface ChatResponse {
  reply: string;
}
