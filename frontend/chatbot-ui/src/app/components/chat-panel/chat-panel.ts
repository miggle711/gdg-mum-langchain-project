import { Component, output, ViewChild, ElementRef, AfterViewChecked, ChangeDetectionStrategy, ChangeDetectorRef, OnInit } from '@angular/core';
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
  changeDetection: ChangeDetectionStrategy.Default,
})
export class ChatPanel implements AfterViewChecked, OnInit {
  closed = output<void>();
  @ViewChild('messageList') private messageList!: ElementRef;

  userMessage = '';
  isLoading = false;
  messages: { role: 'user' | 'assistant'; text: string }[] = [];
  private shouldScroll = false;

  constructor(private chat: Chat, private cdr: ChangeDetectorRef) {}

  ngOnInit() {
    this.initializeConversation();
  }

  private initializeConversation() {
    this.chat.startConversation().subscribe({
      next: (response) => {
        console.log('Chat initialized, conversation ID:', response.conversation_id);
        this.messages = [{ role: 'assistant', text: response.message }];
        this.shouldScroll = true;
        this.cdr.markForCheck();
      },
      error: (error) => {
        console.error('Failed to start conversation:', error);
        this.messages = [{ role: 'assistant', text: 'Failed to start chat. Please refresh and try again.' }];
        this.cdr.markForCheck();
      },
    });
  }

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

    console.log('Sending message:', text);
    console.log('Current messages:', this.messages);

    this.chat.send(text).subscribe({
      next: (response: ChatResponse) => {
        console.log('Received response:', response);
        console.log('Response reply:', response.reply);
        this.messages.push({ role: 'assistant', text: response.reply });
        console.log('Messages after push:', this.messages);
        this.isLoading = false;
        this.shouldScroll = true;
        this.cdr.markForCheck();
      },
      error: (error) => {
        console.log('Error received:', error);
        this.messages.push({
          role: 'assistant',
          text: `Error: ${error.error?.detail || 'Failed to get response from server. Make sure the backend is running.'}`,
        });
        this.isLoading = false;
        this.shouldScroll = true;
        this.cdr.markForCheck();
      },
    });
  }
}
