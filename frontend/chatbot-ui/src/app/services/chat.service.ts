import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface ChatRequest {
  conversation_id: string;
  message: string;
}

export interface ChatResponse {
  conversation_id: string;
  response: string;
  message_count: number;
}

export interface Conversation {
  conversation_id: string;
  history: string;
  message_count: number;
}

export interface ConversationSummary {
  conversation_id: string;
  message_count: number;
}

@Injectable({
  providedIn: 'root',
})
export class ChatService {
  private apiUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  sendMessage(conversationId: string, message: string): Observable<ChatResponse> {
    const request: ChatRequest = {
      conversation_id: conversationId,
      message,
    };
    return this.http.post<ChatResponse>(`${this.apiUrl}/chat`, request);
  }

  getConversation(conversationId: string): Observable<Conversation> {
    return this.http.get<Conversation>(
      `${this.apiUrl}/conversation/${conversationId}`
    );
  }

  deleteConversation(conversationId: string): Observable<any> {
    return this.http.delete(`${this.apiUrl}/conversation/${conversationId}`);
  }

  listConversations(): Observable<{ conversations: ConversationSummary[] }> {
    return this.http.get<{ conversations: ConversationSummary[] }>(
      `${this.apiUrl}/conversations`
    );
  }
}
