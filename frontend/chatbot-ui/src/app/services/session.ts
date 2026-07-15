import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { map } from 'rxjs/operators';

const SESSION_ID_STORAGE_KEY = 'session_id';

@Injectable({
  providedIn: 'root',
})
export class Session {
  private apiUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  // Returns the persisted session_id if one exists in localStorage (no
  // network call — same browser/session recognizes itself instantly),
  // otherwise calls POST /session/start to mint a new one and persists it.
  getOrCreateSessionId(): Observable<string> {
    const stored = localStorage.getItem(SESSION_ID_STORAGE_KEY);
    if (stored) {
      return of(stored);
    }

    return this.http
      .post<{ session_id: string; message: string }>(`${this.apiUrl}/session/start`, {})
      .pipe(
        map((response) => {
          localStorage.setItem(SESSION_ID_STORAGE_KEY, response.session_id);
          return response.session_id;
        })
      );
  }

  getSessionId(): string | null {
    return localStorage.getItem(SESSION_ID_STORAGE_KEY);
  }
}
