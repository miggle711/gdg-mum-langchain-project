import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';


// Injectable means this service can be injected into components
@Injectable({ 
  providedIn: 'root', // This makes the service available application-wide without needing to add it to a module's providers array
})
export class Chat {
  private apiUrl = '/api';  // Base URL for the backend API
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

  send(userPrompt: string, systemPrompt?: string): Observable<ChatResponse> {
    /**
     * This method sends a user message to the backend's /chat endpoint and returns the assistant's reply.
     * 
     * args:
     *   userPrompt: the message from the user to send to the backend
     *   systemPrompt: an optional system prompt that can provide additional context for the conversation (not currently used in this implementation but can be added to the request payload if needed)
     * returns: an Observable that emits the assistant's reply from the backend
     */
    if (!this.conversationId) {
      throw new Error('Conversation not started. Call startConversation() first.');
    }
    console.log('Chat service sending request to:', `${this.apiUrl}/chat`);
    return this.http
      .post<any>(`${this.apiUrl}/chat`, {
        conversation_id: this.conversationId,
        message: userPrompt,
      })
      .pipe( // pipe allows us to transform the response from the backend before it gets to the component that called this method
        map((response) => {
          console.log('Chat service received response:', response);
          return { reply: response.response }; // we return an object with a 'reply' property that contains the assistant's response from the backend, which matches the ChatResponse interface expected by the component
        })
      );
  }

  // Getter and setter for the conversation ID, which can be useful if we want to manage the conversation ID separately or need to reset it when starting a new conversation
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
