import { TestBed } from '@angular/core/testing';

import { Chat } from './chat';

describe('Chat', () => {
  let service: Chat;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(Chat);
  });

  // Test to check if the Chat service is created successfully
  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  // Test to check if the send method returns a mock reply as expected
  it('should return a mock reply', async () => {
    const response = await service.send('hello').toPromise();
    expect(response!.reply).toContain('Mock reply to: hello');
  });
});
