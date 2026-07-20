const state = { containers: [], services: [], traffic: [] };
const elements = {
  containerCount: document.querySelector("#container-count"),
  containerNotice: document.querySelector("#container-notice"),
  containers: document.querySelector("#containers"),
  containersEmpty: document.querySelector("#containers-empty"),
  exportMap: document.querySelector("#export-map"),
  imageCount: document.querySelector("#image-count"),
  refreshAll: document.querySelector("#refresh-all"),
  refreshContainers: document.querySelector("#refresh-containers"),
  refreshServices: document.querySelector("#refresh-services"),
  runningCount: document.querySelector("#running-count"),
  serviceCount: document.querySelector("#service-count"),
  serviceNotice: document.querySelector("#service-notice"),
  services: document.querySelector("#services"),
  servicesEmpty: document.querySelector("#services-empty"),
  stoppedCount: document.querySelector("#stopped-count"),
  startPodman: document.querySelector("#start-podman"),
  toast: document.querySelector("#toast"),
  traffic: document.querySelector("#traffic"),
  trafficCount: document.querySelector("#traffic-count"),
  trafficEmpty: document.querySelector("#traffic-empty")
};

let toastTimer;

const notify = message => {
  clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  toastTimer = setTimeout(() => elements.toast.classList.remove("visible"), 2400);
};

const hostCall = async (action, payload = {}) => {
  const response = await chrome.runtime.sendMessage({ source: "localhost-radar", action, ...payload });
  if (!response?.ok) throw new Error(response?.error || "Native host request failed.");
  return response.data;
};

const setNotice = (element, message) => {
  element.hidden = !message;
  element.textContent = message || "";
};

const makeButton = (label, className, handler, disabled = false) => {
  const button = document.createElement("button");
  button.className = className;
  button.textContent = label;
  button.disabled = disabled;
  button.addEventListener("click", handler);
  return button;
};

const copy = async (value, message) => {
  await navigator.clipboard.writeText(value);
  notify(message);
};

const renderContainers = () => {
  const running = state.containers.filter(container => container.running);
  elements.containerCount.textContent = state.containers.length;
  elements.runningCount.textContent = running.length;
  elements.stoppedCount.textContent = state.containers.length - running.length;
  elements.imageCount.textContent = new Set(state.containers.map(container => container.image)).size;
  elements.containersEmpty.hidden = state.containers.length > 0;
  elements.containers.replaceChildren(...state.containers.map(container => {
    const card = document.createElement("article");
    card.className = `container-card${container.running ? " running" : ""}`;
    const main = document.createElement("div");
    main.className = "card-main";
    const top = document.createElement("div");
    top.className = "card-top";
    const name = document.createElement("h2");
    name.className = "container-name";
    name.textContent = container.name;
    name.title = container.name;
    const status = document.createElement("span");
    status.className = `state${container.running ? " running" : ""}`;
    status.textContent = container.state;
    top.append(name, status);
    const image = document.createElement("p");
    image.className = "image-name";
    image.textContent = container.image;
    image.title = container.image;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${container.id.slice(0, 12)} · ${container.ports || "no published ports"}`;
    main.append(top, image, meta);
    const actions = document.createElement("div");
    actions.className = "card-actions";
    actions.append(
      makeButton("Copy shell", "action", () => copy(LocalhostRadarLib.containerCommand(container), "Container access command copied"), !container.running),
      makeButton("Restart", "action", () => runContainerAction(container, "restart")),
      makeButton("Stop", "action danger", () => runContainerAction(container, "stop"), !container.running)
    );
    card.append(main, actions);
    return card;
  }));
};

const runContainerAction = async (container, action) => {
  if (!confirm(`${action === "stop" ? "Stop" : "Restart"} ${container.name}?`)) return;
  try {
    await hostCall("container_action", { container: container.id, operation: action });
    notify(`${container.name} ${action === "stop" ? "stopped" : "restarted"}`);
    await loadContainers();
  } catch (error) {
    setNotice(elements.containerNotice, error.message);
  }
};

const loadContainers = async () => {
  setNotice(elements.containerNotice, "");
  try {
    state.containers = await hostCall("list_containers");
    renderContainers();
  } catch (error) {
    state.containers = [];
    renderContainers();
    setNotice(elements.containerNotice, error.message);
  }
};

const renderServices = () => {
  elements.serviceCount.textContent = state.services.length;
  elements.servicesEmpty.hidden = state.services.length > 0;
  elements.services.replaceChildren(...state.services.map(service => {
    const row = document.createElement("tr");
    const port = document.createElement("td");
    port.className = "port";
    port.textContent = service.port;
    const process = document.createElement("td");
    process.textContent = `${service.name} · ${service.pid}`;
    const language = document.createElement("td");
    language.textContent = service.language;
    const path = document.createElement("td");
    path.textContent = service.path || "unknown";
    path.title = service.path || "unknown";
    const actionsCell = document.createElement("td");
    const actions = document.createElement("div");
    actions.className = "row-actions";
    actions.append(
      makeButton("Open", "action", () => chrome.tabs.create({ url: LocalhostRadarLib.serviceUrl(service) })),
      makeButton("curl", "action", () => copy(LocalhostRadarLib.curlCommand(LocalhostRadarLib.serviceUrl(service)), "curl command copied")),
      makeButton("Kill", "action danger", () => killService(service), service.protected)
    );
    actionsCell.append(actions);
    row.append(port, process, language, path, actionsCell);
    return row;
  }));
};

const killService = async service => {
  if (!confirm(`Terminate ${service.name} on port ${service.port}?`)) return;
  try {
    await hostCall("kill_process", { pid: service.pid });
    notify(`${service.name} terminated`);
    await loadServices();
  } catch (error) {
    setNotice(elements.serviceNotice, error.message);
  }
};

const loadServices = async () => {
  setNotice(elements.serviceNotice, "");
  try {
    state.services = await hostCall("list_services");
    renderServices();
  } catch (error) {
    state.services = [];
    renderServices();
    setNotice(elements.serviceNotice, error.message);
  }
};

const renderTraffic = () => {
  elements.trafficCount.textContent = `${state.traffic.length} ${state.traffic.length === 1 ? "request" : "requests"}`;
  elements.trafficEmpty.hidden = state.traffic.length > 0;
  elements.traffic.replaceChildren(...state.traffic.slice(0, 100).map(request => {
    const row = document.createElement("article");
    row.className = "request";
    const method = document.createElement("span");
    method.className = "method";
    method.textContent = request.method;
    const status = document.createElement("span");
    status.className = `status ${LocalhostRadarLib.statusTone(request.status)}`;
    status.textContent = request.status || "ERR";
    const url = document.createElement("span");
    url.className = "url";
    url.textContent = request.url;
    url.title = request.url;
    const duration = document.createElement("span");
    duration.className = "duration";
    duration.textContent = LocalhostRadarLib.duration(request.duration);
    const copyButton = makeButton("C", "", () => copy(LocalhostRadarLib.curlCommand(request.url), "curl command copied"));
    copyButton.title = "Copy curl command";
    row.append(method, status, url, duration, copyButton);
    return row;
  }));
};

document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(item => item.classList.toggle("active", item === tab));
    document.querySelectorAll(".view").forEach(view => view.classList.toggle("active", view.id === `${tab.dataset.tab}-view`));
  });
});

elements.refreshContainers.addEventListener("click", loadContainers);
elements.startPodman.addEventListener("click", async () => {
  elements.startPodman.disabled = true;
  elements.startPodman.textContent = "Starting";
  setNotice(elements.containerNotice, "");
  try {
    await hostCall("start_podman_machine");
    notify("Podman machine started");
    await loadContainers();
  } catch (error) {
    setNotice(elements.containerNotice, error.message);
  } finally {
    elements.startPodman.disabled = false;
    elements.startPodman.textContent = "Start / Repair Podman";
  }
});
elements.refreshServices.addEventListener("click", loadServices);
elements.refreshAll.addEventListener("click", () => Promise.all([loadContainers(), loadServices()]));
elements.exportMap.addEventListener("click", () => {
  const payload = JSON.stringify({ capturedAt: new Date().toISOString(), services: state.services, traffic: state.traffic }, null, 2);
  const url = URL.createObjectURL(new Blob([payload], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "localhost-api-map.json";
  link.click();
  URL.revokeObjectURL(url);
  notify("API map saved");
});

if (chrome.devtools?.network?.onRequestFinished) {
  chrome.devtools.network.onRequestFinished.addListener(request => {
    const url = request.request.url;
    if (!/^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(?::\d+)?/i.test(url) && !/^wss?:\/\/(localhost|127\.0\.0\.1|\[::1\])(?::\d+)?/i.test(url)) return;
    state.traffic.unshift({
      duration: request.time,
      method: request.request.method,
      status: request.response.status,
      type: request._resourceType || "request",
      url
    });
    state.traffic = state.traffic.slice(0, 250);
    renderTraffic();
  });
} else {
  elements.trafficEmpty.querySelector("p").textContent = "Open Chrome DevTools and select Localhost Radar to capture request traffic.";
}

renderTraffic();
Promise.all([loadContainers(), loadServices()]);
