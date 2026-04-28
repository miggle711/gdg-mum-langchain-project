import { Component, output, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatDividerModule } from '@angular/material/divider';
import { TextFieldModule } from '@angular/cdk/text-field';
import { Chat, ChatResponse } from '../../services/chat';

@Component({
  selector: 'app-chat-panel',
  imports: [
    FormsModule,
    MatButtonModule,
    MatIconModule,
    MatFormFieldModule,
    MatInputModule,
    MatProgressBarModule,
    MatDividerModule,
    TextFieldModule,
  ],
  templateUrl: './chat-panel.html',
  styleUrl: './chat-panel.scss',
})
export class ChatPanel implements AfterViewChecked {
  closed = output<void>();
  @ViewChild('messageList') private messageList!: ElementRef;

  userMessage = '';
  isLoading = false;
  messages: { role: 'user' | 'assistant'; text: string }[] = [];
  private shouldScroll = false;

  constructor(private chat: Chat) {}

  ngAfterViewChecked() {
    if (this.shouldScroll) {
      const el = this.messageList?.nativeElement;
      if (el) el.scrollTop = el.scrollHeight;
      this.shouldScroll = false;
    }
  }

  onClose() {
    this.closed.emit();
  }

  onEnter(event: Event) {
    if (!(event as KeyboardEvent).shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
  }

  sendMessage() {
    if (!this.userMessage.trim()) return;
    const text = this.userMessage.trim();
    this.messages.push({ role: 'user', text });
    this.userMessage = '';
    this.isLoading = true;
    this.shouldScroll = true;

    this.chat.send(text).subscribe((response: ChatResponse) => {
      this.messages.push({ role: 'assistant', text: response.reply });
      this.isLoading = false;
      this.shouldScroll = true;
    });
  }
}
