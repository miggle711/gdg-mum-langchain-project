import { Injectable } from '@angular/core';
import { Observable, of } from 'rxjs';
import { delay } from 'rxjs/operators';

@Injectable({
  providedIn: 'root',
})
// Chat class to handle communication with the chatbot backend
export class Chat {
  // Mock implementation - replace with actual API call
  // Send a user prompt to the chatbot and receive a response
  send(userPrompt: string, systemPrompt?: string): Observable<ChatResponse> {
    return of({ reply: `Mock reply to: ${userPrompt}` }).pipe(delay(600));
  }
}
// ChatResponse interface to define the structure of the chatbot's response
export interface ChatResponse { reply: string; }
