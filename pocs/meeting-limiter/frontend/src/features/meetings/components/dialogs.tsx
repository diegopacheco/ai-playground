import { createContext, use, useState, type ReactNode } from "react";
import type { Meeting } from "../types";
import { MeetingForm } from "./MeetingForm";
import { MeetingDetails } from "./MeetingDetails";

type Dialog = { readonly kind: "none" } | { readonly kind: "new"; readonly start: Date } | { readonly kind: "details"; readonly meeting: Meeting };

type Dialogs = { readonly openNew: (start?: Date) => void; readonly openDetails: (m: Meeting) => void };

const DialogContext = createContext<Dialogs | undefined>(undefined);

function nextHalfHour(now: Date): Date {
  const d = new Date(now);
  d.setMinutes(d.getMinutes() < 30 ? 30 : 60, 0, 0);
  return d;
}

export function MeetingDialogs({ children }: { readonly children: ReactNode }) {
  const [dialog, setDialog] = useState<Dialog>({ kind: "none" });
  const close = () => setDialog({ kind: "none" });
  const value: Dialogs = {
    openNew: (start) => setDialog({ kind: "new", start: start ?? nextHalfHour(new Date()) }),
    openDetails: (meeting) => setDialog({ kind: "details", meeting }),
  };
  return (
    <DialogContext value={value}>
      {children}
      {dialog.kind === "new" && <MeetingForm start={dialog.start} onClose={close} />}
      {dialog.kind === "details" && <MeetingDetails meeting={dialog.meeting} onClose={close} />}
    </DialogContext>
  );
}

export function useMeetingDialogs(): Dialogs {
  const ctx = use(DialogContext);
  if (!ctx) throw new Error("MeetingDialogs is missing");
  return ctx;
}
