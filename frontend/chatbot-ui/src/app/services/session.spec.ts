import { firstValueFrom } from 'rxjs';
import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';

import { Session } from './session';

describe('Session', () => {
  let service: Session;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    localStorage.clear();
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
    });
    service = TestBed.inject(Session);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
    localStorage.clear();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('returns the persisted session_id without making an HTTP call', async () => {
    localStorage.setItem('session_id', 'existing-session-id');

    const sessionId = await firstValueFrom(service.getOrCreateSessionId());

    expect(sessionId).toBe('existing-session-id');
    httpMock.expectNone('http://localhost:8000/session/start');
  });

  it('calls /session/start and persists the result when nothing is stored', async () => {
    const resultPromise = firstValueFrom(service.getOrCreateSessionId());

    const req = httpMock.expectOne('http://localhost:8000/session/start');
    expect(req.request.method).toBe('POST');
    req.flush({ session_id: 'new-session-id', message: 'Welcome to our store! How can I help you find the perfect product today?' });

    const sessionId = await resultPromise;
    expect(sessionId).toBe('new-session-id');
    expect(localStorage.getItem('session_id')).toBe('new-session-id');
  });

  it('getSessionId returns the persisted value synchronously', () => {
    expect(service.getSessionId()).toBeNull();

    localStorage.setItem('session_id', 'existing-session-id');
    expect(service.getSessionId()).toBe('existing-session-id');
  });
});
