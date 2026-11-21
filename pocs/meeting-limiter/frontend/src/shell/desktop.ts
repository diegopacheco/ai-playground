export type DesktopBridge = {
  readonly toggleMaximize: () => void;
  readonly onToast: (callback: (text: string) => void) => void;
};

declare global {
  interface Window {
    readonly meetingLimiter?: DesktopBridge;
  }
}

export function desktop(): DesktopBridge | undefined {
  return window.meetingLimiter;
}

export function isDesktop(): boolean {
  return window.meetingLimiter !== undefined;
}
