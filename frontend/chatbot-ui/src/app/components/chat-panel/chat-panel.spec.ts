import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ChatPanel } from './chat-panel';

describe('ChatPanel', () => {
  let component: ChatPanel;
  let fixture: ComponentFixture<ChatPanel>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ChatPanel],
    }).compileComponents();

    fixture = TestBed.createComponent(ChatPanel);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
