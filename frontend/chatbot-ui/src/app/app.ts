import { Component } from '@angular/core';
import { ChatButton } from './components/chat-button/chat-button';
import { ChatPanel } from './components/chat-panel/chat-panel';

@Component({
  selector: 'app-root',
  imports: [ChatButton, ChatPanel],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {
  isChatOpen = false;

  openChat() {
    this.isChatOpen = !this.isChatOpen;
  }
}
