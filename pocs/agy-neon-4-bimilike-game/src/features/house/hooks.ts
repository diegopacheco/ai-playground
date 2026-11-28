import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { logger } from "../../lib/logger";
import { browserStorage } from "../../lib/storage";
import { createInitialHouseState, loadHouseState, reduceHouseState, saveHouseState } from "./state";
import type { HouseAction, HouseState, HouseUpdateResult } from "./types";

export const houseKeys = {
  all: ["house"] as const,
  state: () => [...houseKeys.all, "state"] as const,
};

export function useHouseState() {
  return useQuery({
    queryKey: houseKeys.state(),
    queryFn: () => loadHouseState(browserStorage),
    initialData: () => loadHouseState(browserStorage),
  });
}

export function useHouseAction(onUnlock?: (newlyUnlocked: HouseUpdateResult["newlyUnlocked"]) => void) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (action: HouseAction): Promise<HouseUpdateResult> => {
      const current = queryClient.getQueryData<HouseState>(houseKeys.state()) ?? createInitialHouseState();
      const result = reduceHouseState(current, action);
      saveHouseState(browserStorage, result.state);
      logger.info("house.action", { type: action.type, newlyUnlocked: result.newlyUnlocked.length });
      return result;
    },
    onSuccess: (result) => {
      queryClient.setQueryData(houseKeys.state(), result.state);
      if (result.newlyUnlocked.length > 0 && onUnlock) {
        onUnlock(result.newlyUnlocked);
      }
    },
  });
}
