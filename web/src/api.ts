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

// Uploads go through XMLHttpRequest because fetch can't report how much of a big file has been sent.
export function upload<T>(path: string, file: File, onProgress: (fraction: number) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("PUT", path);
    request.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    request.onload = () => {
      if (request.status === 401) loggedOut();
      let data: { detail?: unknown } | null = null;
      try {
        data = JSON.parse(request.responseText);
      } catch {
        data = null;
      }
      if (request.status >= 200 && request.status < 300) resolve(data as T);
      else {
        const detail = data && typeof data.detail === "string" ? data.detail : `The server answered ${request.status}.`;
        reject(new ApiError(request.status, detail));
      }
    };
    request.onerror = () => reject(new ApiError(0, "The upload stopped. Check the connection and try again."));
    request.send(file);
  });
}

export function isAbort(error: unknown) {
  return error instanceof DOMException && error.name === "AbortError";
}
