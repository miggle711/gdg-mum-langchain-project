import { Component, input, output } from '@angular/core';

@Component({
  selector: 'app-chat-panel',
  imports: [],
  templateUrl: './chat-panel.html',
  styleUrl: './chat-panel.scss',
})
export class ChatPanel {
  open = input(false);
  closed = output<void>();

  onClose() {
    this.closed.emit();
  }
}