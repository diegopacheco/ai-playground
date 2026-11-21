import { useForm } from "@tanstack/react-form";
import { Avatar } from "../../../components/Person";
import { formatMinutes, formatMoney } from "../../../money";
import { useActingUser } from "../../../shell/acting";
import { Icon } from "../../../shell/icons";
import { Modal } from "../../../shell/panels";
import { toDateKey, toTimeKey } from "../../../time";
import { useUsers } from "../../people/hooks";
import { quoteFromError } from "../api";
import { useBookMeeting, useQuote } from "../hooks";
import { DURATION_OPTIONS, TIME_OPTIONS, toRequest, type MeetingDraft } from "../request";
import { QuoteTable } from "./QuoteTable";

type Props = { readonly start: Date; readonly onClose: () => void };

type FooterProps = { readonly draft: MeetingDraft; readonly submitting: boolean; readonly onClose: () => void };

function QuoteFooter({ draft, submitting, onClose }: FooterProps) {
  const quote = useQuote(toRequest({ ...draft, title: draft.title || "quote" }));
  const allowed = quote.data?.allowed === true && !quote.isPlaceholderData;
  return (
    <>
      {quote.error && <p className="quote-warning">{quote.error.message}</p>}
      {!quote.error && !quote.data && <p className="muted">Calculating cost...</p>}
      {!quote.error && quote.data && <QuoteTable quote={quote.data} />}
      <div className="form-actions">
        <button type="button" className="button" onClick={onClose}>
          Discard
        </button>
        <button type="submit" className="button button--primary" disabled={submitting || !allowed}>
          <Icon name="coin" />
          {submitting ? "Booking..." : allowed ? `Pay ${formatMoney(quote.data?.totalCost ?? 0)} and send` : "Not enough funds"}
        </button>
      </div>
    </>
  );
}

export function MeetingForm({ start, onClose }: Props) {
  const { userId } = useActingUser();
  const users = useUsers();
  const book = useBookMeeting();
  const form = useForm({
    defaultValues: {
      title: "",
      organizerId: userId,
      attendeeIds: [] as string[],
      date: toDateKey(start),
      time: toTimeKey(start),
      minutes: 60,
    },
    onSubmit: async ({ value }) => {
      const request = toRequest(value);
      if (!request) return;
      await book.mutateAsync(request);
      onClose();
    },
  });

  return (
    <Modal id="new-meeting" title="New meeting" icon={<Icon name="plus" />} onClose={onClose} className="meeting-form">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void form.handleSubmit();
        }}
      >
        <form.Field name="title" validators={{ onSubmit: ({ value }) => (value.trim() === "" ? "Add a title" : undefined) }}>
          {(field) => (
            <label className="field">
              <span>Title</span>
              <input
                autoFocus
                value={field.state.value}
                maxLength={120}
                placeholder="Add a title"
                onChange={(e) => field.handleChange(e.target.value)}
                onBlur={field.handleBlur}
              />
              {field.state.meta.errors.length > 0 && <em className="field-error">{field.state.meta.errors.join(", ")}</em>}
            </label>
          )}
        </form.Field>
        <div className="field-row">
          <form.Field name="date">
            {(field) => (
              <label className="field">
                <span>Date</span>
                <input type="date" value={field.state.value} onChange={(e) => field.handleChange(e.target.value)} />
              </label>
            )}
          </form.Field>
          <form.Field name="time">
            {(field) => (
              <label className="field">
                <span>Start</span>
                <select value={field.state.value} onChange={(e) => field.handleChange(e.target.value)}>
                  {TIME_OPTIONS.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>
              </label>
            )}
          </form.Field>
          <form.Field name="minutes">
            {(field) => (
              <label className="field">
                <span>Duration</span>
                <select value={field.state.value} onChange={(e) => field.handleChange(Number(e.target.value))}>
                  {DURATION_OPTIONS.map((m) => (
                    <option key={m} value={m}>
                      {formatMinutes(m)}
                    </option>
                  ))}
                </select>
              </label>
            )}
          </form.Field>
        </div>
        <form.Field name="organizerId">
          {(field) => (
            <label className="field">
              <span>Organizer</span>
              <select value={field.state.value} onChange={(e) => field.handleChange(e.target.value)}>
                {(users.data ?? []).map((u) => (
                  <option key={u.id} value={u.id}>
                    {u.name} - {u.title}
                  </option>
                ))}
              </select>
            </label>
          )}
        </form.Field>
        <form.Subscribe selector={(s) => s.values.organizerId}>
          {(organizerId) => (
            <form.Field name="attendeeIds">
              {(field) => (
                <fieldset className="field">
                  <legend>Attendees</legend>
                  <div className="chips">
                    {(users.data ?? [])
                      .filter((u) => u.id !== organizerId)
                      .map((u) => {
                        const on = field.state.value.includes(u.id);
                        return (
                          <button
                            key={u.id}
                            type="button"
                            className={`chip ${on ? "chip--on" : ""}`}
                            aria-pressed={on}
                            onClick={() => field.handleChange(on ? field.state.value.filter((id) => id !== u.id) : [...field.state.value, u.id])}
                          >
                            <Avatar user={u} size={22} />
                            {u.name}
                          </button>
                        );
                      })}
                  </div>
                </fieldset>
              )}
            </form.Field>
          )}
        </form.Subscribe>
        <form.Subscribe selector={(s) => [s.values, s.isSubmitting] as const}>
          {([values, submitting]) => <QuoteFooter draft={values} submitting={submitting} onClose={onClose} />}
        </form.Subscribe>
        {book.error && (
          <p className="quote-warning" role="alert">
            {quoteFromError(book.error) ? "Someone ran out of funds while you were booking. Check the costs above." : book.error.message}
          </p>
        )}
      </form>
    </Modal>
  );
}
