import { Component, input, output } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { Chat, ChatResponse } from '../../services/chat';

type Message = {
  role: 'user' | 'assistant';
  text: string;
};

@Component({
  selector: 'app-chat-panel',
  imports: [CommonModule, FormsModule],
  templateUrl: './chat-panel.html',
  styleUrl: './chat-panel.scss',
})
export class ChatPanel {
  open = input(false);
  closed = output<void>();

  userMessage = '';
  isLoading = false;
  messages: Message[] = [];

  constructor(private chat: Chat) {}

  onClose() {
    this.closed.emit();
  }

  sendMessage() {
    if (!this.userMessage.trim()) return;

    const text = this.userMessage.trim();
    this.messages.push({ role: 'user', text });
    this.userMessage = '';
    this.isLoading = true;

    this.chat.send(text).subscribe((response: ChatResponse) => {
      this.messages.push({ role: 'assistant', text: response.reply });
      this.isLoading = false;
    });
  }
}