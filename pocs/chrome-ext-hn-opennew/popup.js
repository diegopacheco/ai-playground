const MAX_TABS = 30;
const FRONT_PAGE = "https://news.ycombinator.com/";

const statusEl = document.getElementById("status");
const listEl = document.getElementById("list");
const buttonEl = document.getElementById("open");
const hintEl = document.getElementById("hint");

function collect(doc, base) {
  const links = doc.querySelectorAll(".athing .titleline > a, .athing a.storylink");
  return [...links]
    .map((a) => {
      const row = a.closest(".athing");
      const rank = row.querySelector(".rank");
      return {
        rank: rank ? rank.textContent.replace(".", "").trim() : "",
        title: a.textContent.trim(),
        url: new URL(a.getAttribute("href"), base).href
      };
    })
    .filter((story) => new URL(story.url).hostname !== "news.ycombinator.com");
}

async function pageToRead() {
  const [tab] = await chrome.tabs.query({
    currentWindow: true,
    active: true,
    url: "https://news.ycombinator.com/*"
  });
  return tab ? tab.url : FRONT_PAGE;
}

async function fetchStories(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(response.status);
  const doc = new DOMParser().parseFromString(await response.text(), "text/html");
  return collect(doc, url);
}

async function grayVisitedTabs() {
  try {
    const tabs = await chrome.tabs.query({ url: "https://news.ycombinator.com/*" });
    for (const tab of tabs) {
      await chrome.scripting.insertCSS({ target: { tabId: tab.id }, files: ["content.css"] });
      await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["content.js"] });
    }
  } catch {}
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
  statusEl.textContent = "Loading Hacker News...";

  let stories;
  try {
    stories = await fetchStories(await pageToRead());
  } catch {
    statusEl.textContent = "Could not reach news.ycombinator.com.";
    hintEl.textContent = "Check your connection and click the icon again.";
    return;
  }

  grayVisitedTabs();

  const unread = (await filterUnvisited(stories)).slice(0, MAX_TABS);

  if (unread.length === 0) {
    statusEl.textContent = `All ${stories.length} stories were already opened.`;
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
