const RESULTS_URL = 'https://www.youtube.com/results?search_query=';

const HEADERS = {
  'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36',
  'Accept-Language': 'en-US,en;q=0.9',
  Cookie: 'SOCS=CAI'
};

export function toSeconds(text) {
  if (!text) return 0;
  return text.split(':').reduce((total, part) => total * 60 + Number(part), 0);
}

function collectVideos(node, found) {
  if (Array.isArray(node)) {
    node.forEach(item => collectVideos(item, found));
  } else if (node && typeof node === 'object') {
    if (node.videoRenderer) found.push(node.videoRenderer);
    else Object.values(node).forEach(value => collectVideos(value, found));
  }
  return found;
}

function toTrack(video) {
  const duration = video.lengthText?.simpleText || '';
  return {
    id: video.videoId,
    title: video.title?.runs?.map(run => run.text).join('') || '',
    channel: video.ownerText?.runs?.[0]?.text || '',
    duration,
    seconds: toSeconds(duration)
  };
}

export function parseResults(html) {
  const match = html.match(/var ytInitialData = (\{.*?\});<\/script>/s);
  if (!match) throw new Error('YouTube results page had no ytInitialData');
  return collectVideos(JSON.parse(match[1]), [])
    .map(toTrack)
    .filter(track => track.id && track.title && track.seconds > 0);
}

export async function search(query) {
  const response = await fetch(RESULTS_URL + encodeURIComponent(query), { headers: HEADERS });
  if (!response.ok) throw new Error(`YouTube answered ${response.status}`);
  return parseResults(await response.text());
}
