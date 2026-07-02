import { Component, output, ViewChild, ElementRef, AfterViewChecked, ChangeDetectionStrategy, ChangeDetectorRef, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatDividerModule } from '@angular/material/divider';
import { TextFieldModule } from '@angular/cdk/text-field';
import { Chat } from '../../services/chat';

/**
 * ChatPanel is an Angular component that provides a user interface for a chatbot conversation. 
 * It allows users to send messages and receive responses from the chatbot, which are displayed in a chat-like format. 
 * The component manages the conversation state, handles user input, and interacts with the Chat service to communicate with the backend API.
 *  */ 

@Component({
  selector: 'app-chat-panel',
  imports: [ // these are the Angular modules that this component depends on for its functionality and UI elements
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

  // Array of message objects
  // Format: { role: 'user' | 'assistant', text: string }
  // Example:
  // messages = [
  //   { role: 'user', text: 'Hello' },
  //   { role: 'assistant', text: 'Hi there!' },
  //   { role: 'user', text: 'How are you?' },
  //   { role: 'assistant', text: 'I\'m doing great!' },
  // ]
  messages: { role: 'user' | 'assistant'; text: string; traceId?: string; feedback?: boolean }[] = [];
  private shouldScroll = false;

  // add the Chat service from the services folder, which is responsible for communicating with the backend API to start conversations and send messages,
  //  as well as ChangeDetectorRef to manually trigger change detection when we update the messages array or loading state
  constructor(private chat: Chat, private cdr: ChangeDetectorRef) {}


  // OnInit is a lifecycle hook that is called after the component is initialized, and we use it to start a new conversation with the chatbot when the component loads
  // a hook is a special method that Angular calls at specific points in the component's lifecycle, allowing us to run custom code when those events occur (like when the component is created, updated, or destroyed)
  ngOnInit() {
    this.initializeConversation();
  }

  private initializeConversation() {
    /**
     * This method initializes a new conversation with the chatbot by calling the startConversation method of the Chat service.
     * It subscribes to the Observable returned by startConversation to handle the asynchronous response from the backend.
     * On success, it sets the initial message from the chatbot and stores the conversation ID in the Chat service for future messages.
     * On error, it displays an error message in the chat panel.
     */

    // We call the startConversation method of the Chat service, which returns an Observable that we subscribe to in order to handle the response from the backend
    // the observable will emit the conversation ID and initial message from the backend when the conversation is successfully started, or an error if something goes wrong
    this.chat.startConversation().subscribe({
      next: (response) => {
        console.log('Chat initialized, conversation ID:', response.conversation_id);

        // We set the initial message from the chatbot in the messages array, which will be displayed in the chat panel.
        this.messages = [{ role: 'assistant', text: response.message }];
        this.shouldScroll = true; 
        this.cdr.markForCheck(); // we manually trigger change detection to ensure the UI updates with the new message and scrolls to the bottom of the chat panel
      },
      error: (error) => {
        console.error('Failed to start conversation:', error);
        this.messages = [{ role: 'assistant', text: 'Failed to start chat. Please refresh and try again.' }];
        this.cdr.markForCheck();
      },
    });
  }

  ngAfterViewChecked() { 
    // After the view has been checked and updated, we check if we need to scroll to the bottom of the chat panel (when a new message is added) and do so if necessary
    if (this.shouldScroll) {
      const el = this.messageList?.nativeElement;
      if (el) el.scrollTop = el.scrollHeight;
      this.shouldScroll = false;
    }
  }

  onClose() {
    // When the user clicks the close button, we emit the closed event to notify the parent component that the chat panel should be closed
    this.closed.emit();
  }

  onEnter(event: Event) {
    // This method is called when the user presses the Enter key in the message input field. If Shift is not held, it prevents the default behavior (which would be to add a new line) and calls the sendMessage method to send the user's message to the chatbot.
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
    this.cdr.markForCheck();

    // Push a placeholder for the assistant reply that we'll fill in token by token
    this.messages.push({ role: 'assistant', text: '' });
    const assistantIndex = this.messages.length - 1;

    this.chat.sendStream(
      text,
      (traceId) => {
        this.messages[assistantIndex].traceId = traceId;
        this.cdr.markForCheck();
      },
      (token) => {
        this.messages[assistantIndex].text += token;
        this.shouldScroll = true;
        this.cdr.markForCheck();
      },
      (_fullText) => {
        this.isLoading = false;
        this.shouldScroll = true;
        this.cdr.markForCheck();
      },
      (err) => {
        this.messages[assistantIndex].text = `Error: ${err}`;
        this.isLoading = false;
        this.shouldScroll = true;
        this.cdr.markForCheck();
      },
    );
  }

  rateFeedback(index: number, value: boolean) {
    const msg = this.messages[index];
    if (!msg.traceId || msg.feedback !== undefined) return; // no trace yet, or already rated
    msg.feedback = value;
    this.cdr.markForCheck();
    this.chat.sendFeedback(msg.traceId, value).subscribe({
      error: (err) => {
        console.error('Failed to send feedback', err);
        msg.feedback = undefined; // allow retry on failure
        this.cdr.markForCheck();
      },
    });
  }
}
