import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { tap } from 'rxjs/operators';

const ACCESS_TOKEN_STORAGE_KEY = 'access_token';

interface AuthResponse {
  access_token: string;
  token_type: string;
  user_id: number;
}

// Injectable means this service can be injected into components
@Injectable({
  providedIn: 'root',
})
export class Auth {
  private apiUrl = '/api';

  constructor(private http: HttpClient) {}

  // Persists the JWT on success (signup doubles as login server-side —
  // see backend/app/routes/auth.py), so callers don't need a separate
  // login step right after signing up.
  signup(email: string, password: string, name: string): Observable<AuthResponse> {
    return this.http
      .post<AuthResponse>(`${this.apiUrl}/auth/signup`, { email, password, name })
      .pipe(tap((response) => this.storeToken(response.access_token)));
  }

  login(email: string, password: string): Observable<AuthResponse> {
    return this.http
      .post<AuthResponse>(`${this.apiUrl}/auth/login`, { email, password })
      .pipe(tap((response) => this.storeToken(response.access_token)));
  }

  logout(): void {
    localStorage.removeItem(ACCESS_TOKEN_STORAGE_KEY);
  }

  getToken(): string | null {
    return localStorage.getItem(ACCESS_TOKEN_STORAGE_KEY);
  }

  isLoggedIn(): boolean {
    return this.getToken() !== null;
  }

  // Builds the Authorization header for an authenticated request, or an
  // empty object if logged out — spread this into any HttpClient call's
  // headers so guest requests are unaffected (backend/session_identity.py's
  // resolve_user and app/routes/chat.py's _conversation_key both fall back
  // to session_id-based guest identity when no header is present).
  authHeader(): { Authorization: string } | {} {
    const token = this.getToken();
    return token ? { Authorization: `Bearer ${token}` } : {};
  }

  private storeToken(token: string): void {
    localStorage.setItem(ACCESS_TOKEN_STORAGE_KEY, token);
  }
}
