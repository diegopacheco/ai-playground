import { useState } from "react";
import { sound } from "../../../lib/sound";
import { CHARACTERS, getCharacterSpec } from "../../characters/catalog";
import type { CharacterId, CharacterPlacement } from "../../characters/types";
import { replyToTopic, replyToVoice, TOPIC_BUTTONS } from "../dialogue";
import type { TalkTopic } from "../types";

type PhoneModalProps = {
  readonly unlockedIds: readonly CharacterId[];
  readonly onCalled: (characterId: CharacterId, mood: CharacterPlacement["mood"]) => void;
  readonly onClose: () => void;
};

type SpeechRecognitionLike = {
  lang: string;
  interimResults: boolean;
  maxAlternatives: number;
  onresult: ((ev: { readonly results: { readonly [idx: number]: { readonly [idx: number]: { readonly transcript: string } } } }) => void) | null;
  onerror: (() => void) | null;
  onend: (() => void) | null;
  start: () => void;
};

declare global {
  interface Window {
    SpeechRecognition?: new () => SpeechRecognitionLike;
    webkitSpeechRecognition?: new () => SpeechRecognitionLike;
  }
}

function createSpeechRecognizer(): SpeechRecognitionLike | null {
  if (typeof window === "undefined") return null;
  const Ctor = window.SpeechRecognition ?? window.webkitSpeechRecognition;
  if (!Ctor) return null;
  try {
    return new Ctor();
  } catch {
    return null;
  }
}

export function PhoneModal({ unlockedIds, onCalled, onClose }: PhoneModalProps) {
  const [activeFriend, setActiveFriend] = useState<CharacterId | null>(null);
  const [ringing, setRinging] = useState(false);
  const [bubble, setBubble] = useState("Tap a friend to call them!");
  const [reaction, setReaction] = useState("📞");
  const [listening, setListening] = useState(false);

  const unlockedChars = CHARACTERS.filter((c) => unlockedIds.includes(c.id));
  const currentSpec = activeFriend ? getCharacterSpec(activeFriend) : null;

  const startCall = (id: CharacterId) => {
    const spec = getCharacterSpec(id);
    setActiveFriend(id);
    setRinging(true);
    setReaction("🔔");
    setBubble(`Calling ${spec.name}...`);
    sound.ring();
    window.setTimeout(() => {
      setRinging(false);
      const reply = replyToTopic(id, "hi");
      setBubble(reply.text);
      setReaction(reply.reactionEmoji);
      sound.chirp(spec.chirpMultiplier);
      sound.speak(reply.text, spec.voicePitch, spec.voiceRate);
      onCalled(id, reply.mood);
    }, 680);
  };

  const handleTopic = (topic: TalkTopic) => {
    if (!activeFriend || ringing) return;
    const spec = getCharacterSpec(activeFriend);
    const reply = replyToTopic(activeFriend, topic);
    setBubble(reply.text);
    setReaction(reply.reactionEmoji);
    sound.chirp(spec.chirpMultiplier);
    sound.speak(reply.text, spec.voicePitch, spec.voiceRate);
    onCalled(activeFriend, reply.mood);
  };

  const handleVoiceTalk = () => {
    if (!activeFriend || ringing || listening) return;
    const spec = getCharacterSpec(activeFriend);
    const recognizer = createSpeechRecognizer();
    if (!recognizer) {
      const reply = replyToVoice(activeFriend, "");
      setBubble(reply.text);
      setReaction(reply.reactionEmoji);
      sound.chirp(spec.chirpMultiplier);
      sound.speak(reply.text, spec.voicePitch, spec.voiceRate);
      onCalled(activeFriend, reply.mood);
      return;
    }

    setListening(true);
    setBubble("Listening! Say hi!");
    setReaction("🎤");
    recognizer.lang = "en-US";
    recognizer.interimResults = false;
    recognizer.maxAlternatives = 1;

    let handled = false;
    const finish = (spoken: string) => {
      if (handled) return;
      handled = true;
      setListening(false);
      const reply = replyToVoice(activeFriend, spoken);
      setBubble(reply.text);
      setReaction(reply.reactionEmoji);
      sound.chirp(spec.chirpMultiplier);
      sound.speak(reply.text, spec.voicePitch, spec.voiceRate);
      onCalled(activeFriend, reply.mood);
    };

    recognizer.onresult = (ev) => {
      const transcript = ev.results[0]?.[0]?.transcript ?? "";
      finish(transcript);
    };
    recognizer.onerror = () => finish("");
    recognizer.onend = () => finish("");
    try {
      recognizer.start();
    } catch {
      finish("");
    }
  };

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="Toy Phone">
      <div className="toy-modal phone-modal">
        <div className="modal-top">
          <h2 className="modal-title">📞 Toy Phone</h2>
          <button type="button" className="toy-close" onClick={onClose} aria-label="Hang up phone">
            ✕
          </button>
        </div>

        <div className="phone-screen" style={{ backgroundColor: currentSpec ? `${currentSpec.bodyColor}33` : "#eef9ff" }}>
          <div className={`phone-avatar ${ringing ? "avatar-ring" : "avatar-bounce"}`}>
            <span>{currentSpec ? currentSpec.emoji : "📱"}</span>
          </div>
          <div className="phone-reaction">{reaction}</div>
          <p className="phone-bubble">{bubble}</p>
        </div>

        {!activeFriend ? (
          <div className="phone-contacts">
            {unlockedChars.map((char) => (
              <button
                key={char.id}
                type="button"
                className="contact-btn"
                style={{ backgroundColor: char.bodyColor }}
                onClick={() => startCall(char.id)}
              >
                <span className="contact-emoji">{char.emoji}</span>
                <span className="contact-name">{char.name}</span>
              </button>
            ))}
          </div>
        ) : (
          <div className="phone-talk-area">
            <div className="topic-grid">
              {TOPIC_BUTTONS.map((btn) => (
                <button
                  key={btn.id}
                  type="button"
                  className="topic-btn"
                  style={{ backgroundColor: btn.color }}
                  onClick={() => handleTopic(btn.id)}
                  disabled={ringing}
                >
                  <span className="topic-emoji">{btn.emoji}</span>
                  <span>{btn.label}</span>
                </button>
              ))}
            </div>
            <div className="phone-bottom-row">
              <button
                type="button"
                className={`toy-big-btn toy-btn-mic ${listening ? "mic-live" : ""}`}
                onClick={handleVoiceTalk}
                disabled={ringing}
              >
                {listening ? "🎤 Listening..." : "🎤 Talk!"}
              </button>
              <button
                type="button"
                className="toy-big-btn toy-btn-hangup"
                onClick={() => {
                  sound.pop();
                  setActiveFriend(null);
                  setBubble("Tap a friend to call them!");
                  setReaction("📞");
                }}
              >
                🔄 Friends
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
