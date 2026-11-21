import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { addDays } from "../../time";
import { logger } from "../../logger";
import { walletKeys } from "../wallets/hooks";
import { rankingKeys } from "../rankings/hooks";
import { meetingsApi } from "./api";
import type { MeetingRequest } from "./types";

export const meetingKeys = {
  all: ["meetings"] as const,
  week: (weekStart: Date) => [...meetingKeys.all, "week", weekStart.toISOString()] as const,
  quoteAll: ["quote"] as const,
  quote: (r: MeetingRequest) =>
    [...meetingKeys.quoteAll, r.organizerId, [...r.attendeeIds].sort().join(","), r.start.toISOString(), r.end.toISOString()] as const,
};

export function useWeekMeetings(weekStart: Date) {
  return useQuery({
    queryKey: meetingKeys.week(weekStart),
    queryFn: () => meetingsApi.list(weekStart, addDays(weekStart, 7)),
    placeholderData: keepPreviousData,
  });
}

export function useQuote(request: MeetingRequest | undefined) {
  return useQuery({
    queryKey: request ? meetingKeys.quote(request) : meetingKeys.quoteAll,
    queryFn: () => meetingsApi.quote(request ?? fail()),
    enabled: request !== undefined,
    placeholderData: keepPreviousData,
    retry: false,
  });
}

function fail(): never {
  throw new Error("quote requested without a meeting");
}

function useRefreshMoney() {
  const client = useQueryClient();
  return () =>
    Promise.all(
      [meetingKeys.all, meetingKeys.quoteAll, walletKeys.all, rankingKeys.all].map((queryKey) => client.invalidateQueries({ queryKey })),
    );
}

export function useBookMeeting() {
  const refresh = useRefreshMoney();
  return useMutation({
    mutationFn: meetingsApi.book,
    onSuccess: (m) => {
      logger.info("meeting booked", { meetingId: m.id, totalCost: m.totalCost });
      return refresh();
    },
  });
}

export function useCancelMeeting() {
  const refresh = useRefreshMoney();
  return useMutation({
    mutationFn: meetingsApi.cancel,
    onSuccess: (_, id) => {
      logger.info("meeting cancelled", { meetingId: id });
      return refresh();
    },
  });
}
