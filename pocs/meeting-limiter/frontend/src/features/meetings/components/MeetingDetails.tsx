import { PersonCell } from "../../../components/Person";
import { formatMinutes, formatMoney } from "../../../money";
import { Icon } from "../../../shell/icons";
import { Modal } from "../../../shell/panels";
import { formatWhen } from "../../../time";
import { useUserMap } from "../../people/hooks";
import { useCancelMeeting } from "../hooks";
import { durationMinutes, type Meeting } from "../types";

type Props = { readonly meeting: Meeting; readonly onClose: () => void };

export function MeetingDetails({ meeting, onClose }: Props) {
  const users = useUserMap();
  const cancel = useCancelMeeting();
  const organizer = users.get(meeting.organizerId);
  return (
    <Modal id="meeting-details" title={meeting.title} icon={<Icon name="calendar" />} onClose={onClose} className="meeting-details">
      <p className="details-when">
        <Icon name="clock" />
        {formatWhen(meeting.start, meeting.end)} ({formatMinutes(durationMinutes(meeting))})
      </p>
      <p className="details-when">
        <Icon name="user" />
        Organized by {organizer?.name ?? meeting.organizerId}
      </p>
      <div className="quote-total ok">
        <span>Total cost for {meeting.charges.length} people</span>
        <strong>{formatMoney(meeting.totalCost)}</strong>
      </div>
      <table className="data-table compact">
        <thead>
          <tr>
            <th>Person</th>
            <th>Paid</th>
          </tr>
        </thead>
        <tbody>
          {meeting.charges.map((c) => {
            const user = users.get(c.userId);
            return (
              <tr key={c.userId}>
                <td>{user ? <PersonCell user={user} /> : c.userId}</td>
                <td>{formatMoney(c.amount)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {cancel.error && <p className="quote-warning">{cancel.error.message}</p>}
      <div className="form-actions">
        <button type="button" className="button" onClick={onClose}>
          Close
        </button>
        <button type="button" className="button button--danger" disabled={cancel.isPending} onClick={() => cancel.mutate(meeting.id, { onSuccess: onClose })}>
          <Icon name="trash" />
          Cancel meeting and refund
        </button>
      </div>
    </Modal>
  );
}
