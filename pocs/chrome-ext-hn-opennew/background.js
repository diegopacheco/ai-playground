chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type !== "visited") return;
  Promise.all(
    message.urls.map(async (url) => {
      try {
        const visits = await chrome.history.getVisits({ url });
        return visits.length > 0;
      } catch {
        return false;
      }
    })
  ).then(sendResponse);
  return true;
});
