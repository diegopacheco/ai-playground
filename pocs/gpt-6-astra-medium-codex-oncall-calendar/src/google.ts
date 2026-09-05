import type { Shift } from "./main";

type TokenResponse = { access_token?: string; error?: string };
type GoogleIdentity = {
  accounts: {
    oauth2: {
      initTokenClient: (options: {
        client_id: string;
        scope: string;
        callback: (response: TokenResponse) => void;
        error_callback: () => void;
      }) => { requestAccessToken: () => void };
    };
  };
};
declare global {
  interface Window {
    google?: GoogleIdentity;
  }
}

export function loadGoogle(): Promise<void> {
  if (window.google) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://accounts.google.com/gsi/client";
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("Google sign-in could not load."));
    document.head.appendChild(script);
  });
}

export async function importToGoogle(
  clientId: string,
  events: Shift[],
  timezone: string,
  progress: (message: string) => void,
): Promise<void> {
  const google = window.google;
  if (!google)
    throw new Error("Google sign-in is still loading. Please try again.");
  const token = await new Promise<string>((resolve, reject) => {
    google.accounts.oauth2
      .initTokenClient({
        client_id: clientId,
        scope: "https://www.googleapis.com/auth/calendar.events.owned",
        callback: (response) =>
          response.access_token
            ? resolve(response.access_token)
            : reject(
                new Error(
                  "Google authorization was declined. No shifts were added.",
                ),
              ),
        error_callback: () =>
          reject(
            new Error(
              "Google sign-in was closed or blocked. Please try again.",
            ),
          ),
      })
      .requestAccessToken();
  });
  let created = 0;
  let skipped = 0;
  for (const event of events) {
    try {
      const response = await fetch(
        "https://www.googleapis.com/calendar/v3/calendars/primary/events",
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            id: event.id,
            summary: event.summary,
            start: { dateTime: event.start, timeZone: timezone },
            end: { dateTime: event.end, timeZone: timezone },
          }),
          signal: AbortSignal.timeout(30000),
        },
      );
      if (response.status === 409) skipped++;
      else if (response.ok) created++;
      else throw new Error(`Google returned ${response.status}.`);
      progress(
        `${created + skipped}/${events.length} shifts processed · ${created} added · ${skipped} already present`,
      );
    } catch (cause) {
      throw new Error(
        `${created} shifts added, ${skipped} already present. ${cause instanceof Error ? cause.message : "Connection failed."} Retry to continue; existing events will be skipped.`,
      );
    }
  }
  progress(
    `All set. ${created} shifts added to Google Calendar; ${skipped} already present.`,
  );
}
