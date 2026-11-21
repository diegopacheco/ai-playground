# Changelog

Records what changed in Meeting Limiter for its users.

## 1.0.0 - 2026-11-20

### Added
- Week calendar in the Outlook 365 style where clicking an empty slot opens the New meeting form for that time.
- Weekly wallets per person: engineers get 5 hours at $100/h, managers 10 hours at $150/h, directors and VPs 15 hours at $250/h.
- Every meeting charges the organizer and each attendee for its length at their role rate, and shows its total cost.
- The New meeting form shows a live quote with each person's cost and balance before and after.
- Booking is refused when any participant would go below $0, the quote names who is short, and "Pay and send" turns into "Not enough funds".
- Wallets refill every Monday, so 6 hours of meetings ruin an engineer's week until the next one.
- Cancelling a meeting refunds every participant.
- Meetings tab with every meeting of the week and its cost.
- Wallets tab with budget, spent, balance and hours used per person, plus week totals and the count of empty wallets.
- Rankings tab with the most expensive meetings, who spends the most money and who spends the most time in meetings, for this week, this month or all time.
- "Acting as" picker in the title bar to choose the organizer.
- Command palette on ⌘K, shortcuts modal on ⌘/, tabs on ⌘1 to ⌘4, new meeting on ⌥N, previous and next week on ⌥← and ⌥→, this week on ⌥T.
- Every panel and modal can expand to almost full screen or hide into a docked tab on the nearest edge, and hidden panels stay hidden after a restart.
- Tooltips with the action name and shortcut keys on hover.
- macOS app with a boot screen that starts the services, stops them on quit, allows a single instance, remembers its window position, maximizes on title bar double click, and saves a screenshot to the Desktop on ⌘P.
- Phone layout with one day at a time, a bottom tab bar and 44px touch targets.
- Scripts to set up, start, stop, check, test, open, install and uninstall the app.

### Verified
- `./scripts/test-all.sh` passed: 14 Go tests (28 with subtests) and 19 Bun tests, 0 failed, 0 skipped, plus gofmt, go vet and the TypeScript type check.
- Against the running API: an engineer with 5 booked hours had a $0 balance, a 6th hour got 402, and the next week showed a full $500 wallet.
- In headless Chromium: "Pay and send" was disabled when a participant was short, a hidden panel came back from its docked tab, Esc collapsed an expanded panel, and there were no console errors.
- On a 440px wide phone viewport the Calendar and Rankings tabs had no horizontal scroll.
- The installed macOS app started both services on launch, stayed at one main process when opened twice, and stopped both services on quit.
- `start-all.sh` and `stop-all.sh` both succeeded when run twice in a row.
