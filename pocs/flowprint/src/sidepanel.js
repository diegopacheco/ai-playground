const elements = {
  clear: document.querySelector("#clear"),
  code: document.querySelector("#code"),
  copy: document.querySelector("#copy"),
  download: document.querySelector("#download"),
  empty: document.querySelector("#empty"),
  record: document.querySelector("#record"),
  report: document.querySelector("#report"),
  run: document.querySelector("#run"),
  status: document.querySelector("#status"),
  stepCount: document.querySelector("#step-count"),
  steps: document.querySelector("#steps"),
  toast: document.querySelector("#toast")
};

let state = { recording: false, steps: [] };
let execution = { reportAvailable: false, running: false };
let toastTimer;
const runnerUrl = "http://127.0.0.1:17339";

const notify = message => {
  clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  toastTimer = setTimeout(() => elements.toast.classList.remove("visible"), 1800);
};

const currentTab = async () => {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs[0];
};

const renderExecution = () => {
  elements.run.disabled = !state.steps.length || state.recording || execution.running;
  elements.run.textContent = execution.running ? "Running Playwright..." : "Run Playwright";
  elements.report.disabled = !execution.reportAvailable || execution.running;
  elements.status.classList.toggle("running", execution.running);
  if (execution.running) elements.status.querySelector("strong").textContent = "RUNNING";
};

const renderCode = source => {
  const nodes = FlowPrintLib.highlightTest(source).map(token => {
    if (token.type === "plain") return document.createTextNode(token.value);
    const span = document.createElement("span");
    span.className = `token-${token.type}`;
    span.textContent = token.value;
    return span;
  });
  elements.code.replaceChildren(...nodes);
};

const render = nextState => {
  state = nextState || { recording: false, steps: [] };
  elements.record.textContent = state.recording ? "Stop recording" : "Start recording";
  elements.record.classList.toggle("live", state.recording);
  elements.status.classList.toggle("live", state.recording);
  elements.status.querySelector("strong").textContent = state.recording ? "RECORDING" : "READY";
  elements.stepCount.textContent = `${state.steps.length} ${state.steps.length === 1 ? "mark" : "marks"}`;
  elements.empty.hidden = state.steps.length > 0;
  elements.steps.replaceChildren(...state.steps.map((step, index) => {
    const description = FlowPrintLib.describeStep(step);
    const item = document.createElement("li");
    item.className = "step";
    const number = document.createElement("span");
    number.className = "step-index";
    number.textContent = String(index + 1).padStart(2, "0");
    const label = document.createElement("span");
    label.className = "step-label";
    label.textContent = description.label;
    const detail = document.createElement("span");
    detail.className = "step-detail";
    detail.textContent = description.detail;
    detail.title = description.detail;
    item.append(number, label, detail);
    return item;
  }));
  renderCode(state.steps.length ? FlowPrintLib.generateTest(state) : "Start a recording to create a test.");
  renderExecution();
};

elements.record.addEventListener("click", async () => {
  if (state.recording) {
    render(await chrome.runtime.sendMessage({ type: "flowprint:stop" }));
    return;
  }
  const tab = await currentTab();
  if (!tab?.id || !/^https?:/.test(tab.url || "")) {
    notify("Open an HTTP page first");
    return;
  }
  render(await chrome.runtime.sendMessage({ type: "flowprint:start", tabId: tab.id, url: tab.url, title: tab.title }));
});

elements.clear.addEventListener("click", async () => {
  if (state.recording) await chrome.runtime.sendMessage({ type: "flowprint:stop" });
  render(await chrome.runtime.sendMessage({ type: "flowprint:clear" }));
});

elements.copy.addEventListener("click", async () => {
  if (!state.steps.length) return notify("Nothing to copy yet");
  await navigator.clipboard.writeText(FlowPrintLib.generateTest(state));
  notify("Test copied");
});

elements.download.addEventListener("click", () => {
  if (!state.steps.length) return notify("Nothing to save yet");
  const url = URL.createObjectURL(new Blob([FlowPrintLib.generateTest(state)], { type: "text/typescript" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "flow.spec.ts";
  link.click();
  URL.revokeObjectURL(url);
  notify("Test saved");
});

elements.run.addEventListener("click", async () => {
  if (!state.steps.length) return notify("Nothing to run yet");
  execution.running = true;
  renderExecution();
  try {
    const response = await fetch(`${runnerUrl}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec: FlowPrintLib.generateTest(state) })
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Playwright could not run");
    execution.reportAvailable = result.reportAvailable;
    notify(result.passed ? "Playwright run passed" : "Playwright run failed");
  } catch (error) {
    notify(error instanceof TypeError ? "Run ./start-runner.sh first" : error.message);
  } finally {
    execution.running = false;
    render(state);
  }
});

elements.report.addEventListener("click", async () => {
  if (!execution.reportAvailable) return notify("Run Playwright first");
  await chrome.tabs.create({ url: `${runnerUrl}/report/` });
});

chrome.runtime.onMessage.addListener(message => {
  if (message.type === "flowprint:state-changed") render(message.state);
});

Promise.all([
  chrome.runtime.sendMessage({ type: "flowprint:get-state" }),
  fetch(`${runnerUrl}/status`).then(response => response.json()).catch(() => ({ reportAvailable: false }))
]).then(([storedState, runner]) => {
  execution.reportAvailable = Boolean(runner.reportAvailable);
  execution.running = Boolean(runner.running);
  render(storedState);
});
