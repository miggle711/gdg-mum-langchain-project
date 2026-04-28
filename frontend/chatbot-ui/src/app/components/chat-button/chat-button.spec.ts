import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ChatButton } from './chat-button';

describe('ChatButton', () => {
  let component: ChatButton;
  let fixture: ComponentFixture<ChatButton>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ChatButton],
    }).compileComponents();

    fixture = TestBed.createComponent(ChatButton);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
