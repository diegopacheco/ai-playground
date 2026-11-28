import { useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { config } from "../lib/config";
import { setSoundEnabled, sound } from "../lib/sound";
import { CHARACTERS, getCharacterSpec } from "../features/characters/catalog";
import { UnlockToast } from "../features/characters/components/UnlockToast";
import type { CharacterId, CharacterPlacement } from "../features/characters/types";
import { ColorPaletteBar } from "../features/customize/components/ColorPaletteBar";
import { useHouseAction, useHouseState } from "../features/house/hooks";
import { DollhouseCanvas } from "../features/house/scene/DollhouseCanvas";
import type { RoomFocus, SelectionTarget, SurfaceId, ViewMode } from "../features/house/types";
import { KitchenSheet } from "../features/kitchen/components/KitchenSheet";
import { getDishSpec } from "../features/kitchen/recipes";
import type { DishSpec } from "../features/kitchen/types";
import { PhoneModal } from "../features/phone/components/PhoneModal";

const ROOM_TABS: readonly { readonly id: RoomFocus; readonly label: string; readonly emoji: string }[] = [
  { id: "all", label: "House", emoji: "🏠" },
  { id: "kitchen", label: "Kitchen", emoji: "🍳" },
  { id: "living", label: "Living", emoji: "🛋️" },
  { id: "bedroom", label: "Bedroom", emoji: "🛏️" },
];

export function HousePage() {
  const navigate = useNavigate();
  const { data: house } = useHouseState();
  const [roomFocus, setRoomFocus] = useState<RoomFocus>("all");
  const [viewMode, setViewMode] = useState<ViewMode>("3d");
  const [soundOn, setSoundOn] = useState<boolean>(config.defaultSoundOn);
  const [selection, setSelection] = useState<SelectionTarget | null>(null);
  const [kitchenOpen, setKitchenOpen] = useState(false);
  const [phoneOpen, setPhoneOpen] = useState(false);
  const [feedingDishUid, setFeedingDishUid] = useState<string | null>(null);
  const [unlockBanner, setUnlockBanner] = useState<CharacterId | null>(null);
  const [chatBanner, setChatBanner] = useState<string | null>(null);
  const [burstTick, setBurstTick] = useState(0);

  const action = useHouseAction((newlyUnlocked) => {
    const first = newlyUnlocked[0];
    if (first) {
      sound.unlockFanfare();
      setUnlockBanner(first);
      setBurstTick((n) => n + 1);
    }
  });

  const showChat = (msg: string) => {
    setChatBanner(msg);
  };

  const handleToggleSound = () => {
    const next = !soundOn;
    setSoundOn(next);
    setSoundEnabled(next);
    if (next) sound.pop();
  };

  const handleTapFurnitureSpecial = (id: string) => {
    sound.pop();
    if (id === "stove") {
      setRoomFocus("kitchen");
      setKitchenOpen(true);
      return;
    }
    if (id === "phoneTable") {
      setPhoneOpen(true);
      return;
    }
    if (id === "lamp" || id === "fridge") {
      action.mutate({ type: "toggleFurnitureActive", id });
    }
  };

  const handleTapCharacter = (id: CharacterId) => {
    const spec = getCharacterSpec(id);
    if (feedingDishUid) {
      sound.munch();
      action.mutate({
        type: "feedCharacter",
        dishUid: feedingDishUid,
        characterId: id,
        now: Date.now(),
      });
      setFeedingDishUid(null);
      showChat(`${spec.emoji} ${spec.name}: "Yum yum! Thank you!"`);
      return;
    }
    sound.chirp(spec.chirpMultiplier);
    sound.speak(spec.greeting, spec.voicePitch, spec.voiceRate);
    showChat(`${spec.emoji} ${spec.name}: "${spec.greeting}"`);
  };

  const handleCooked = (dish: DishSpec) => {
    action.mutate({
      type: "cookDish",
      dishId: dish.id,
      uid: `dish-${Date.now()}`,
    });
    setBurstTick((n) => n + 1);
    showChat(`${dish.emoji} Fresh ${dish.name}! Tap it to feed a friend!`);
  };

  const handleCalled = (characterId: CharacterId, mood: CharacterPlacement["mood"]) => {
    action.mutate({
      type: "recordCall",
      characterId,
      mood,
      now: Date.now(),
    });
  };

  const handleQuickPaint = () => {
    sound.pop();
    if (selection && selection.kind !== "character") {
      setSelection(null);
      return;
    }
    const surfaceByRoom: Record<RoomFocus, SurfaceId> = {
      all: "roof",
      kitchen: "kitchenWall",
      living: "livingWall",
      bedroom: "bedroomWall",
    };
    setSelection({ kind: "surface", id: surfaceByRoom[roomFocus] });
  };

  return (
    <div className="game-shell">
      <header className="top-hud">
        <div className="brand-pill">
          <span className="brand-bear">🐻</span>
          <span className="brand-title">Momo&apos;s House</span>
        </div>

        <nav className="room-tabs" aria-label="House Rooms">
          {ROOM_TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              className={`room-tab ${roomFocus === tab.id ? "room-tab-active" : ""}`}
              onClick={() => {
                sound.pop();
                setRoomFocus(tab.id);
              }}
            >
              <span>{tab.emoji}</span>
              <span className="room-tab-text">{tab.label}</span>
            </button>
          ))}
        </nav>

        <div className="top-right-tools">
          <button
            type="button"
            className="toy-pill-btn"
            onClick={() => {
              sound.pop();
              setViewMode((v) => (v === "3d" ? "2d" : "3d"));
            }}
            aria-label="Toggle 3D or 2D house view"
          >
            {viewMode === "3d" ? "🧊 3D" : "🖼️ 2D"}
          </button>
          <button
            type="button"
            className="toy-pill-btn"
            onClick={handleToggleSound}
            aria-label={soundOn ? "Mute sound" : "Enable sound"}
          >
            {soundOn ? "🔊" : "🔇"}
          </button>
        </div>
      </header>

      <main className="stage-wrap">
        <DollhouseCanvas
          state={house}
          roomFocus={roomFocus}
          viewMode={viewMode}
          selection={selection}
          feedingDishUid={feedingDishUid}
          burstTick={burstTick}
          onSelect={setSelection}
          onMoveFurniture={(id, x, z) => action.mutate({ type: "moveFurniture", id, x, z })}
          onMoveCharacter={(id, x, z) => action.mutate({ type: "moveCharacter", id, x, z })}
          onTapFurnitureSpecial={handleTapFurnitureSpecial}
          onTapCharacter={handleTapCharacter}
        />

        {chatBanner && (
          <div className="speech-ribbon" onClick={() => setChatBanner(null)}>
            {chatBanner}
          </div>
        )}

        {house.dishes.length > 0 && (
          <div className="food-tray" aria-label="Cooked Dishes">
            <span className="food-tray-label">
              {feedingDishUid ? "👇 Tap a friend to feed!" : "🍽️ Yummy Food:"}
            </span>
            <div className="food-tray-items">
              {house.dishes.map((d) => {
                const spec = getDishSpec(d.dishId);
                const selected = feedingDishUid === d.uid;
                return (
                  <button
                    key={d.uid}
                    type="button"
                    className={`dish-chip ${selected ? "dish-chip-selected" : ""}`}
                    onClick={() => {
                      sound.pop();
                      setFeedingDishUid(selected ? null : d.uid);
                    }}
                  >
                    <span>{spec.emoji}</span>
                    <span>{spec.name}</span>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {selection && selection.kind !== "character" && (
          <ColorPaletteBar
            selection={selection}
            furniture={house.furniture}
            onPaintFurniture={(id, color) => action.mutate({ type: "paintFurniture", id, color })}
            onCycleStyle={(id) => action.mutate({ type: "cycleFurnitureStyle", id })}
            onPaintSurface={(id, color) => action.mutate({ type: "paintSurface", id, color })}
            onOpenKitchen={() => {
              setSelection(null);
              setKitchenOpen(true);
            }}
            onOpenPhone={() => {
              setSelection(null);
              setPhoneOpen(true);
            }}
            onClose={() => setSelection(null)}
          />
        )}

        {unlockBanner && (
          <UnlockToast characterId={unlockBanner} onDismiss={() => setUnlockBanner(null)} />
        )}
      </main>

      <footer className="bottom-dock" aria-label="Play Controls">
        <button
          type="button"
          className="dock-btn dock-kitchen"
          onClick={() => {
            sound.pop();
            setRoomFocus("kitchen");
            setKitchenOpen(true);
          }}
        >
          <span className="dock-icon">🍳</span>
          <span className="dock-label">Kitchen</span>
        </button>

        <button
          type="button"
          className="dock-btn dock-phone"
          onClick={() => {
            sound.pop();
            setPhoneOpen(true);
          }}
        >
          <span className="dock-icon">📞</span>
          <span className="dock-label">Phone</span>
        </button>

        <button type="button" className="dock-btn dock-paint" onClick={handleQuickPaint}>
          <span className="dock-icon">🎨</span>
          <span className="dock-label">Colors</span>
        </button>

        <button
          type="button"
          className="dock-btn dock-friends"
          onClick={() => {
            sound.pop();
            void navigate({ to: "/friends" });
          }}
        >
          <span className="dock-icon">🌟</span>
          <span className="dock-label">
            Friends {house.unlockedCharacters.length}/{CHARACTERS.length}
          </span>
        </button>

        <button
          type="button"
          className="dock-btn dock-reset"
          onClick={() => {
            sound.pop();
            setSelection(null);
            setFeedingDishUid(null);
            action.mutate({ type: "reset" });
            showChat("✨ Fresh house ready to decorate!");
          }}
          aria-label="Reset house"
        >
          <span className="dock-icon">🔄</span>
          <span className="dock-label">New</span>
        </button>
      </footer>

      {kitchenOpen && (
        <KitchenSheet onCooked={handleCooked} onClose={() => setKitchenOpen(false)} />
      )}

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
