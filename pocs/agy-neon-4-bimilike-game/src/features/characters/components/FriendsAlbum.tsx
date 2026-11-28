import { CHARACTERS } from "../catalog";
import type { ActivityStats, CharacterId } from "../types";
import { getUnlockProgress } from "../unlocks";

type FriendsAlbumProps = {
  readonly unlockedIds: readonly CharacterId[];
  readonly stats: ActivityStats;
  readonly onBackHome: () => void;
  readonly onCallFriend: (id: CharacterId) => void;
};

export function FriendsAlbum({
  unlockedIds,
  stats,
  onBackHome,
  onCallFriend,
}: FriendsAlbumProps) {
  return (
    <main className="friends-page">
      <div className="friends-header">
        <div>
          <h1 className="friends-title">🌟 My Cute Friends</h1>
          <p className="friends-subtitle">
            {unlockedIds.length} of {CHARACTERS.length} friends live in your house!
          </p>
        </div>
        <button type="button" className="toy-big-btn toy-btn-home" onClick={onBackHome}>
          🏠 Back to House
        </button>
      </div>

      <div className="stats-strip">
        <div className="stat-pill">🍳 Cooked: {stats.cookedCount}</div>
        <div className="stat-pill">🎨 Painted: {stats.paintedCount}</div>
        <div className="stat-pill">🛋️ Moved: {stats.movedCount}</div>
        <div className="stat-pill">📞 Calls: {stats.calledCount}</div>
        <div className="stat-pill">😋 Fed: {stats.fedCount}</div>
      </div>

      <div className="friends-grid">
        {CHARACTERS.map((char) => {
          const unlocked = unlockedIds.includes(char.id);
          const progress = getUnlockProgress(char, stats);
          const pct = Math.round((progress.current / progress.target) * 100);
          return (
            <article
              key={char.id}
              className={`friend-card ${unlocked ? "friend-unlocked" : "friend-locked"}`}
              style={{ borderColor: unlocked ? char.bodyColor : "#d8d4e8" }}
            >
              <div
                className="friend-badge"
                style={{ backgroundColor: unlocked ? char.bodyColor : "#ebe7f5" }}
              >
                <span className="friend-emoji">{unlocked ? char.emoji : "🔒"}</span>
              </div>
              <h2 className="friend-name">{char.name}</h2>
              <p className="friend-note">
                {unlocked ? `Loves ${char.favoriteFood}` : char.unlockHint}
              </p>
              {!unlocked ? (
                <div className="progress-track" aria-label={`Progress ${progress.current} of ${progress.target}`}>
                  <div className="progress-fill" style={{ width: `${pct}%` }} />
                  <span className="progress-label">
                    {progress.current} / {progress.target}
                  </span>
                </div>
              ) : (
                <button
                  type="button"
                  className="toy-chip toy-chip-warm"
                  onClick={() => onCallFriend(char.id)}
                >
                  📞 Call {char.name}
                </button>
              )}
            </article>
          );
        })}
      </div>
    </main>
  );
}
