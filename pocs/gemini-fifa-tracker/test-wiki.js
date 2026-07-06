import https from 'https';

const getSearchImg = (query) => new Promise(resolve => {
  const url = `https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch=${encodeURIComponent(query)}&gsrlimit=1&prop=pageimages&pithumbsize=500&format=json`;
  https.get(url, { headers: { 'User-Agent': 'FifaTracker/1.0 (test@example.com)' } }, res => {
    let d = '';
    res.on('data', c => d += c);
    res.on('end', () => {
      try {
        const pages = JSON.parse(d).query.pages;
        const pageId = Object.keys(pages)[0];
        resolve(pages[pageId].thumbnail ? pages[pageId].thumbnail.source : null);
      } catch (e) {
        console.log('Error parsing or no image for', query);
        resolve(null);
      }
    });
  }).on('error', (e) => {
    console.log('HTTP Error:', e.message);
    resolve(null);
  });
});

async function run() {
  console.log(await getSearchImg('Lionel Messi'));
  console.log(await getSearchImg('Tacos food'));
}
run();
