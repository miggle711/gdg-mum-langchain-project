import { Component, signal } from '@angular/core';
import { ChatButton } from './components/chat-button/chat-button';
import { ChatPanel } from './components/chat-panel/chat-panel';

@Component({
  selector: 'app-root',
  imports: [ChatButton, ChatPanel],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {
  isChatOpen = signal(false);

  openChat() {
    this.isChatOpen.set(true);
  }

  closeChat() {
    this.isChatOpen.set(false);
  }
}