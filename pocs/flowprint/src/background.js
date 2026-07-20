const emptyState = () => ({
  recording: false,
  tabId: null,
  startedAt: null,
  title: "",
  steps: []
});

const readState = async () => {
  const stored = await chrome.storage.local.get("flowprintState");
  return stored.flowprintState || emptyState();
};

const writeState = async state => {
  await chrome.storage.local.set({ flowprintState: state });
  chrome.runtime.sendMessage({ type: "flowprint:state-changed", state }).catch(() => {});
  return state;
};

const appendStep = async (step, tabId) => {
  const state = await readState();
  if (!state.recording || state.tabId !== tabId) return state;
  const previous = state.steps[state.steps.length - 1];
  if (step.type === "navigation" && previous?.type === "navigation" && previous.url === step.url) return state;
  state.steps.push({ ...step, id: crypto.randomUUID() });
  return writeState(state);
};

chrome.runtime.onInstalled.addListener(() => {
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  const handle = async () => {
    if (message.type === "flowprint:get-state") return readState();
    if (message.type === "flowprint:ready") {
      const state = await readState();
      return { recording: state.recording && state.tabId === sender.tab?.id };
    }
    if (message.type === "flowprint:start") {
      const state = emptyState();
      state.recording = true;
      state.tabId = message.tabId;
      state.startedAt = Date.now();
      state.title = message.title || "Recorded flow";
      state.steps.push({
        id: crypto.randomUUID(),
        type: "navigation",
        url: message.url,
        timestamp: Date.now()
      });
      await chrome.tabs.sendMessage(message.tabId, { type: "flowprint:set-recording", recording: true }).catch(() => {});
      return writeState(state);
    }
    if (message.type === "flowprint:stop") {
      const state = await readState();
      state.recording = false;
      if (state.tabId !== null) {
        await chrome.tabs.sendMessage(state.tabId, { type: "flowprint:set-recording", recording: false }).catch(() => {});
      }
      return writeState(state);
    }
    if (message.type === "flowprint:clear") return writeState(emptyState());
    if (message.type === "flowprint:record") return appendStep(message.step, sender.tab?.id);
    return null;
  };
  handle().then(sendResponse).catch(error => sendResponse({ error: error.message }));
  return true;
});
