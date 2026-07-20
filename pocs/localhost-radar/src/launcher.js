const status = document.querySelector("#status");
const openButton = document.querySelector("#open");

const openRadar = () => {
  chrome.tabs.create({ url: chrome.runtime.getURL("src/panel.html") }, tab => {
    if (chrome.runtime.lastError || !tab) {
      status.textContent = chrome.runtime.lastError?.message || "Could not open Radar";
      openButton.hidden = false;
      return;
    }
    window.close();
  });
};

openButton.addEventListener("click", openRadar);
openRadar();
