(() => {
  const links = [...document.querySelectorAll(".athing .titleline > a, .athing a.storylink")].filter((a) => {
    try {
      return new URL(a.href).origin !== location.origin;
    } catch {
      return false;
    }
  });

  if (links.length === 0) return;

  chrome.runtime.sendMessage({ type: "visited", urls: links.map((a) => a.href) }, (visited) => {
    if (chrome.runtime.lastError || !visited) return;
    links.forEach((a, index) => a.classList.toggle("hn-open-new-visited", visited[index]));
  });
})();
