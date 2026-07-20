const hostName = "com.diegopacheco.localhost_radar";
const allowedActions = new Set(["list_containers", "container_action", "list_services", "kill_process", "start_podman_machine"]);

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.source !== "localhost-radar" || !allowedActions.has(message.action)) return false;
  chrome.runtime.sendNativeMessage(hostName, message, response => {
    if (chrome.runtime.lastError) {
      sendResponse({ ok: false, error: `${chrome.runtime.lastError.message}. Run ./install.sh and restart Chrome.` });
      return;
    }
    sendResponse(response || { ok: false, error: "Native host returned no data." });
  });
  return true;
});
