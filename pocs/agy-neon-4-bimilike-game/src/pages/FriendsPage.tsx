import { useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { FriendsAlbum } from "../features/characters/components/FriendsAlbum";
import type { CharacterId, CharacterPlacement } from "../features/characters/types";
import { useHouseAction, useHouseState } from "../features/house/hooks";
import { PhoneModal } from "../features/phone/components/PhoneModal";
import { sound } from "../lib/sound";

export function FriendsPage() {
  const navigate = useNavigate();
  const { data: house } = useHouseState();
  const action = useHouseAction();
  const [phoneOpen, setPhoneOpen] = useState(false);

  const handleCallFriend = (_id: CharacterId) => {
    sound.pop();
    setPhoneOpen(true);
  };

  const handleCalled = (characterId: CharacterId, mood: CharacterPlacement["mood"]) => {
    action.mutate({
      type: "recordCall",
      characterId,
      mood,
      now: Date.now(),
    });
  };

  return (
    <div className="friends-shell">
      <FriendsAlbum
        unlockedIds={house.unlockedCharacters}
        stats={house.stats}
        onBackHome={() => {
          sound.pop();
          void navigate({ to: "/" });
        }}
        onCallFriend={handleCallFriend}
      />
      {phoneOpen && (
        <PhoneModal
          unlockedIds={house.unlockedCharacters}
          onCalled={handleCalled}
          onClose={() => setPhoneOpen(false)}
        />
      )}
    </div>
  );
}
