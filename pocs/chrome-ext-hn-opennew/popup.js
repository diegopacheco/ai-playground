const MAX_TABS = 30;

const statusEl = document.getElementById("status");
const listEl = document.getElementById("list");
const buttonEl = document.getElementById("open");
const hintEl = document.getElementById("hint");

function collect() {
  const links = document.querySelectorAll(".athing .titleline > a, .athing a.storylink");
  return [...links]
    .map((a) => {
      const row = a.closest(".athing");
      const rank = row.querySelector(".rank");
      return {
        rank: rank ? rank.textContent.replace(".", "").trim() : "",
        title: a.textContent.trim(),
        url: a.href
      };
    })
    .filter((story) => {
      try {
        return new URL(story.url).origin !== location.origin;
      } catch {
        return false;
      }
    });
}

async function findHackerNewsTab() {
  const tabs = await chrome.tabs.query({
    currentWindow: true,
    url: "https://news.ycombinator.com/*"
  });
  return tabs[0];
}

async function readStories(tabId) {
  const [injection] = await chrome.scripting.executeScript({
    target: { tabId },
    func: collect
  });
  return injection.result || [];
}

async function isVisited(url) {
  try {
    const visits = await chrome.history.getVisits({ url });
    return visits.length > 0;
  } catch {
    return false;
  }
}

async function filterUnvisited(stories) {
  const visited = await Promise.all(stories.map((story) => isVisited(story.url)));
  return stories.filter((story, index) => !visited[index]);
}

function render(stories) {
  listEl.replaceChildren();
  for (const story of stories) {
    const item = document.createElement("li");
    const rank = document.createElement("span");
    rank.className = "rank";
    rank.textContent = story.rank ? `${story.rank}. ` : "";
    const host = document.createElement("span");
    host.className = "host";
    host.textContent = ` (${new URL(story.url).hostname.replace(/^www\./, "")})`;
    item.append(rank, document.createTextNode(story.title), host);
    listEl.append(item);
  }
}

function openTabs(stories) {
  for (const story of stories) {
    chrome.tabs.create({ url: story.url, active: false });
  }
}

async function start() {
  const tab = await findHackerNewsTab();
  if (!tab) {
    statusEl.textContent = "No Hacker News tab in this window.";
    hintEl.textContent = "Open news.ycombinator.com and click the icon again.";
    return;
  }

  const stories = await readStories(tab.id);
  const unread = (await filterUnvisited(stories)).slice(0, MAX_TABS);

  if (unread.length === 0) {
    statusEl.textContent = `All ${stories.length} stories on this page were already opened.`;
    return;
  }

  statusEl.textContent = `${unread.length} of ${stories.length} stories never opened:`;
  render(unread);
  buttonEl.disabled = false;
  buttonEl.textContent = `Open ${unread.length} tabs`;
  hintEl.textContent = "Story links only, comments are never opened.";
  buttonEl.addEventListener("click", () => {
    openTabs(unread);
    window.close();
  });
}

start();
