import unittest
from datetime import date, datetime, timedelta

from schedule import Settings, calendar_file, generate


class ScheduleTests(unittest.TestCase):
    def settings(self, **changes: str | int) -> Settings:
        settings: Settings = {"primary": "2026-09-07", "secondary": "2026-09-14", "interval": 4, "timezone": "America/Los_Angeles"}
        if "primary" in changes:
            settings["primary"] = str(changes["primary"])
        if "secondary" in changes:
            settings["secondary"] = str(changes["secondary"])
        if "interval" in changes:
            settings["interval"] = int(changes["interval"])
        if "timezone" in changes:
            settings["timezone"] = str(changes["timezone"])
        return settings

    def test_each_role_repeats_after_four_weeks_with_seven_daily_shifts(self) -> None:
        schedule = generate(self.settings(), date(2026, 9, 5))
        primary = [event for event in schedule.events if event.role == "Primary"]
        self.assertEqual([event.date for event in primary[:7]], [(date(2026, 9, 7) + timedelta(days=day)).isoformat() for day in range(7)])
        self.assertEqual(primary[7].date, "2026-10-05")
        self.assertEqual(schedule.years, [2026, 2027])
        self.assertTrue(all("2026-09-05" <= event.date <= "2027-12-31" for event in schedule.events))
        self.assertTrue(any(event.date.startswith("2027") for event in schedule.events))
        self.assertEqual(primary[0].summary, "🔥 On Call Primary 07/09-13/09 week")

    def test_active_week_keeps_remaining_days_without_past_events(self) -> None:
        schedule = generate(self.settings(), date(2026, 9, 10))
        self.assertEqual(schedule.events[0].date, "2026-09-10")
        self.assertEqual(schedule.events[0].week_start, "2026-09-07")

    def test_dst_preserves_local_four_to_six_hours(self) -> None:
        schedule = generate(self.settings(), date(2026, 9, 5))
        offsets = {datetime.fromisoformat(event.start).utcoffset() for event in schedule.events}
        self.assertEqual(offsets, {timedelta(hours=-7), timedelta(hours=-8)})
        for event in schedule.events:
            self.assertEqual(datetime.fromisoformat(event.start).hour, 4)
            self.assertEqual(datetime.fromisoformat(event.end).hour, 6)

    def test_overlapping_roles_are_rejected_even_on_later_cycles(self) -> None:
        for secondary in ["2026-09-07", "2026-09-10", "2026-10-06"]:
            with self.subTest(secondary=secondary), self.assertRaisesRegex(ValueError, "overlap"):
                generate(self.settings(secondary=secondary), date(2026, 9, 5))

    def test_invalid_dates_intervals_and_timezones_are_rejected(self) -> None:
        for changes in [{"primary": "invalid"}, {"primary": "2026-02-30"}, {"interval": 1}, {"interval": 53}, {"timezone": "Mars/Orbit"}, {"primary": "2028-01-01"}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                generate(self.settings(**changes), date(2026, 9, 5))

    def test_year_boundary_clips_days_and_preserves_week_label(self) -> None:
        schedule = generate(self.settings(primary="2027-12-29", secondary="2027-12-22"), date(2026, 12, 30))
        primary = [event for event in schedule.events if event.role == "Primary"]
        self.assertEqual(len(primary), 3)
        self.assertEqual(primary[-1].date, "2027-12-31")
        self.assertEqual(primary[-1].week_end, "2028-01-04")

    def test_leap_day_is_scheduled(self) -> None:
        schedule = generate(self.settings(primary="2028-02-28", secondary="2028-03-06"), date(2027, 9, 5))
        self.assertIn("2028-02-29", [event.date for event in schedule.events])

    def test_calendar_has_one_event_per_day_and_stable_unique_ids(self) -> None:
        schedule = generate(self.settings(), date(2026, 9, 5))
        content = calendar_file(schedule)
        self.assertEqual(content.count("BEGIN:VEVENT"), len(schedule.events))
        self.assertEqual(len({event.id for event in schedule.events}), len(schedule.events))
        self.assertEqual(schedule.events, generate(self.settings(), date(2026, 9, 5)).events)
        self.assertIn("DTSTART:20260907T110000Z\r\nDTEND:20260907T130000Z", content)
        self.assertTrue(all(len(line.encode()) <= 75 for line in content.split("\r\n")))
        self.assertTrue(content.endswith("END:VCALENDAR\r\n"))


if __name__ == "__main__":
    unittest.main()
