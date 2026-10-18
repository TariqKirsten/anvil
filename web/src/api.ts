// Calls to the anvil server. Every call sends the login cookie; a 401 anywhere sends the screen back to the login page.

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

let loggedOut: () => void = () => {};

export function onLoggedOut(handler: () => void) {
  loggedOut = handler;
}

export async function api<T>(path: string, options: { method?: string; json?: unknown; signal?: AbortSignal } = {}) {
  const response = await fetch(path, {
    method: options.method ?? (options.json === undefined ? "GET" : "POST"),
    headers: options.json === undefined ? undefined : { "Content-Type": "application/json" },
    body: options.json === undefined ? undefined : JSON.stringify(options.json),
    credentials: "same-origin",
    signal: options.signal,
  });
  if (response.status === 401 && path !== "/api/login") loggedOut();
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data && typeof data.detail === "string" ? data.detail : `The server answered ${response.status}.`;
    throw new ApiError(response.status, detail);
  }
  return data as T;
}

export function isAbort(error: unknown) {
  return error instanceof DOMException && error.name === "AbortError";
}
