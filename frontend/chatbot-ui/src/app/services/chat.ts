import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

@Injectable({
  providedIn: 'root',
})
export class Chat {
  private apiUrl = '/api';
  private conversationId: string = '';

  constructor(private http: HttpClient) {}

  startConversation(): Observable<{ conversation_id: string; message: string }> {
    console.log('Starting new conversation...');
    return this.http.post<any>(`${this.apiUrl}/chat/start`, {}).pipe(
      map((response) => {
        console.log('Conversation started with ID:', response.conversation_id);
        this.conversationId = response.conversation_id;
        return response;
      })
    );
  }

  send(userPrompt: string, systemPrompt?: string): Observable<ChatResponse> {
    if (!this.conversationId) {
      throw new Error('Conversation not started. Call startConversation() first.');
    }
    console.log('Chat service sending request to:', `${this.apiUrl}/chat`);
    return this.http
      .post<any>(`${this.apiUrl}/chat`, {
        conversation_id: this.conversationId,
        message: userPrompt,
      })
      .pipe(
        map((response) => {
          console.log('Chat service received response:', response);
          return { reply: response.response };
        })
      );
  }

  getConversationId(): string {
    return this.conversationId;
  }

  setConversationId(id: string): void {
    this.conversationId = id;
  }
}

export interface ChatResponse {
  reply: string;
}
