import fs from 'fs';
import https from 'https';

const teams = {
  canada: { player: 'Alphonso Davies', dish: 'Poutine' },
  mexico: { player: 'Edson Álvarez', dish: 'Tacos' },
  usa: { player: 'Christian Pulisic', dish: 'Hamburger' },
  austria: { player: 'David Alaba', dish: 'Wiener Schnitzel' },
  belgium: { player: 'Kevin De Bruyne', dish: 'Moules-frites' },
  bosnia: { player: 'Edin Džeko', dish: 'Ćevapi' },
  croatia: { player: 'Luka Modrić', dish: 'Sarma' },
  czechia: { player: 'Patrik Schick', dish: 'Vepřo knedlo zelo' },
  england: { player: 'Harry Kane', dish: 'Fish and chips' },
  france: { player: 'Kylian Mbappé', dish: 'Pot-au-Feu' },
  germany: { player: 'Jamal Musiala', dish: 'Bratwurst' },
  greece: { player: 'Kostas Tsimikas', dish: 'Moussaka' },
  hungary: { player: 'Dominik Szoboszlai', dish: 'Goulash' },
  israel: { player: 'Manor Solomon', dish: 'Falafel' },
  italy: { player: 'Gianluigi Donnarumma', dish: 'Pizza' },
  netherlands: { player: 'Virgil van Dijk', dish: 'Stroopwafel' },
  poland: { player: 'Robert Lewandowski', dish: 'Pierogi' },
  portugal: { player: 'Cristiano Ronaldo', dish: 'Bacalhau' },
  romania: { player: 'Radu Drăgușin', dish: 'Sarmale' },
  scotland: { player: 'Andrew Robertson', dish: 'Haggis' },
  serbia: { player: 'Dušan Vlahović', dish: 'Pljeskavica' },
  slovakia: { player: 'Milan Škriniar', dish: 'Bryndzové halušky' },
  slovenia: { player: 'Jan Oblak', dish: 'Idrijski žlikrofi' },
  spain: { player: 'Pedri', dish: 'Paella' },
  switzerland: { player: 'Granit Xhaka', dish: 'Fondue' },
  turkey: { player: 'Hakan Çalhanoğlu', dish: 'Kebab' },
  ukraine: { player: 'Oleksandr Zinchenko', dish: 'Borscht' },
  wales: { player: 'Brennan Johnson', dish: 'Cawl' },
  egypt: { player: 'Mohamed Salah', dish: 'Koshary' },
  brazil: { player: 'Vinícius Júnior', dish: 'Feijoada' },
  argentina: { player: 'Lionel Messi', dish: 'Asado' },
  colombia: { player: 'Luis Díaz', dish: 'Bandeja Paisa' },
  paraguay: { player: 'Miguel Almirón', dish: 'Sopa Paraguaya' },
  norway: { player: 'Erling Haaland', dish: 'Fårikål' },
  morocco: { player: 'Achraf Hakimi', dish: 'Couscous' },
  japan: { player: 'Kaoru Mitoma', dish: 'Sushi' },
  southkorea: { player: 'Son Heung-min', dish: 'Kimchi' },
  australia: { player: 'Mathew Ryan', dish: 'Meat pie' },
  saudiarabia: { player: 'Salem Al-Dawsari', dish: 'Kabsa' }
};

const getImageUrl = (title) => {
  return new Promise((resolve) => {
    const url = `https://en.wikipedia.org/w/api.php?action=query&titles=${encodeURIComponent(title)}&prop=pageimages&pithumbsize=500&format=json`;
    https.get(url, { headers: { 'User-Agent': 'FifaTrackerBot/1.0' } }, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          const pages = parsed.query.pages;
          const pageId = Object.keys(pages)[0];
          resolve(pages[pageId].thumbnail ? pages[pageId].thumbnail.source : null);
        } catch (e) { resolve(null); }
      });
    }).on('error', () => resolve(null));
  });
};

async function run() {
  const results = {};
  for (const [id, info] of Object.entries(teams)) {
    console.log(`Fetching images for ${id}...`);
    const playerImg = await getImageUrl(info.player);
    const dishImg = await getImageUrl(info.dish);
    results[id] = { playerImg, dishImg };
  }
  fs.writeFileSync('real-images.json', JSON.stringify(results, null, 2));
  console.log('Done.');
}
run();
