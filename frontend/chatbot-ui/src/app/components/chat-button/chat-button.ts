import { Component, output } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';

@Component({
  selector: 'app-chat-button',
  imports: [MatButtonModule, MatIconModule],
  templateUrl: './chat-button.html',
  styleUrl: './chat-button.scss',
})
export class ChatButton {
  clicked = output<void>();

  onClick() {
    this.clicked.emit();
  }
}