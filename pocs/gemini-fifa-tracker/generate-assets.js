import fs from 'fs';
import path from 'path';

const teams = [
  { id: 'canada', name: 'Canada', flag: '🇨🇦', color1: '#ef4444', color2: '#ffffff', players: ['Atiba Hutchinson', 'Craig Forrest', 'Dwayne De Rosario'], star: 'Alphonso Davies', dishes: ['Poutine', 'Butter Tarts', 'Tourtière'] },
  { id: 'mexico', name: 'Mexico', flag: '🇲🇽', color1: '#15803d', color2: '#ef4444', players: ['Hugo Sánchez', 'Rafael Márquez', 'Javier Hernández'], star: 'Santiago Giménez', dishes: ['Tacos', 'Mole Poblano', 'Chiles en Nogada'] },
  { id: 'usa', name: 'USA', flag: '🇺🇸', color1: '#1e3a8a', color2: '#ef4444', players: ['Landon Donovan', 'Clint Dempsey', 'Cobi Jones'], star: 'Christian Pulisic', dishes: ['Hamburger', 'Apple Pie', 'Clam Chowder'] },
  { id: 'austria', name: 'Austria', flag: '🇦🇹', color1: '#dc2626', color2: '#ffffff', players: ['David Alaba', 'Toni Polster', 'Hans Krankl'], star: 'Konrad Laimer', dishes: ['Wiener Schnitzel', 'Sachertorte', 'Apfelstrudel'] },
  { id: 'belgium', name: 'Belgium', flag: '🇧🇪', color1: '#facc15', color2: '#dc2626', players: ['Eden Hazard', 'Vincent Kompany', 'Paul Van Himst'], star: 'Kevin De Bruyne', dishes: ['Moules-Frites', 'Belgian Waffles', 'Carbonnade Flamande'] },
  { id: 'bosnia', name: 'Bosnia and Herzegovina', flag: '🇧🇦', color1: '#1d4ed8', color2: '#facc15', players: ['Edin Džeko', 'Miralem Pjanić', 'Sergej Barbarez'], star: 'Edin Džeko', dishes: ['Ćevapi', 'Burek', 'Klepe'] },
  { id: 'croatia', name: 'Croatia', flag: '🇭🇷', color1: '#dc2626', color2: '#ffffff', players: ['Luka Modrić', 'Davor Šuker', 'Zvonimir Boban'], star: 'Luka Modrić', dishes: ['Peka', 'Crni Rižot', 'Fritule'] },
  { id: 'czechia', name: 'Czechia', flag: '🇨🇿', color1: '#1e3a8a', color2: '#dc2626', players: ['Pavel Nedvěd', 'Petr Čech', 'Josef Masopust'], star: 'Patrik Schick', dishes: ['Vepřo Knedlo Zelo', 'Svíčková', 'Trdelník'] },
  { id: 'england', name: 'England', flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', color1: '#ffffff', color2: '#dc2626', players: ['Bobby Charlton', 'Gary Lineker', 'Bobby Moore'], star: 'Jude Bellingham', dishes: ['Fish and Chips', 'Sunday Roast', 'Shepherd\'s Pie'] },
  { id: 'france', name: 'France', flag: '🇫🇷', color1: '#1d4ed8', color2: '#dc2626', players: ['Zinedine Zidane', 'Michel Platini', 'Thierry Henry'], star: 'Kylian Mbappé', dishes: ['Coq au Vin', 'Ratatouille', 'Crème Brûlée'] },
  { id: 'germany', name: 'Germany', flag: '🇩🇪', color1: '#111827', color2: '#dc2626', players: ['Franz Beckenbauer', 'Gerd Müller', 'Miroslav Klose'], star: 'Florian Wirtz', dishes: ['Bratwurst', 'Sauerkraut', 'Pretzel'] },
  { id: 'netherlands', name: 'Netherlands', flag: '🇳🇱', color1: '#f97316', color2: '#ffffff', players: ['Johan Cruyff', 'Marco van Basten', 'Ruud Gullit'], star: 'Virgil van Dijk', dishes: ['Stroopwafel', 'Bitterballen', 'Stamppot'] },
  { id: 'norway', name: 'Norway', flag: '🇳🇴', color1: '#dc2626', color2: '#1d4ed8', players: ['Erik Thorstvedt', 'John Carew', 'Tore André Flo'], star: 'Erling Haaland', dishes: ['Fårikål', 'Gravlaks', 'Lutefisk'] },
  { id: 'portugal', name: 'Portugal', flag: '🇵🇹', color1: '#16a34a', color2: '#dc2626', players: ['Cristiano Ronaldo', 'Eusébio', 'Luís Figo'], star: 'Bruno Fernandes', dishes: ['Bacalhau à Brás', 'Pastel de Nata', 'Caldo Verde'] },
  { id: 'scotland', name: 'Scotland', flag: '🏴󠁧󠁢󠁳󠁣󠁴󠁿', color1: '#1d4ed8', color2: '#ffffff', players: ['Kenny Dalglish', 'Denis Law', 'Graeme Souness'], star: 'Andrew Robertson', dishes: ['Haggis', 'Scotch Pie', 'Cranachan'] },
  { id: 'spain', name: 'Spain', flag: '🇪🇸', color1: '#dc2626', color2: '#eab308', players: ['Andres Iniesta', 'Xavi Hernandez', 'Iker Casillas'], star: 'Lamine Yamal', dishes: ['Paella', 'Tortilla Española', 'Gazpacho'] },
  { id: 'sweden', name: 'Sweden', flag: '🇸🇪', color1: '#1d4ed8', color2: '#eab308', players: ['Zlatan Ibrahimović', 'Henrik Larsson', 'Gunnar Nordahl'], star: 'Alexander Isak', dishes: ['Köttbullar', 'Gravlax', 'Smörgåstårta'] },
  { id: 'switzerland', name: 'Switzerland', flag: '🇨🇭', color1: '#dc2626', color2: '#ffffff', players: ['Stephane Chapuisat', 'Alexander Frei', 'Xherdan Shaqiri'], star: 'Granit Xhaka', dishes: ['Fondue', 'Raclette', 'Rösti'] },
  { id: 'turkiye', name: 'Türkiye', flag: '🇹🇷', color1: '#dc2626', color2: '#ffffff', players: ['Hakan Şükür', 'Rüştü Reçber', 'Tugay Kerimoğlu'], star: 'Hakan Çalhanoğlu', dishes: ['Kebab', 'Baklava', 'Pide'] },
  { id: 'argentina', name: 'Argentina', flag: '🇦🇷', color1: '#38bdf8', color2: '#ffffff', players: ['Diego Maradona', 'Lionel Messi', 'Mario Kempes'], star: 'Lionel Messi', dishes: ['Asado', 'Empanadas', 'Dulce de Leche'] },
  { id: 'brazil', name: 'Brazil', flag: '🇧🇷', color1: '#eab308', color2: '#16a34a', players: ['Pelé', 'Ronaldo', 'Ronaldinho'], star: 'Vinícius Júnior', dishes: ['Feijoada', 'Pão de Queijo', 'Brigadeiro'] },
  { id: 'colombia', name: 'Colombia', flag: '🇨🇴', color1: '#eab308', color2: '#1d4ed8', players: ['Carlos Valderrama', 'Radamel Falcao', 'Faustino Asprilla'], star: 'Luis Díaz', dishes: ['Bandeja Paisa', 'Arepas', 'Ajiaco'] },
  { id: 'ecuador', name: 'Ecuador', flag: '🇪🇨', color1: '#eab308', color2: '#1d4ed8', players: ['Alex Aguinaga', 'Antonio Valencia', 'Enner Valencia'], star: 'Moisés Caicedo', dishes: ['Ceviche', 'Llapingachos', 'Locro de Papa'] },
  { id: 'paraguay', name: 'Paraguay', flag: '🇵🇾', color1: '#dc2626', color2: '#1e3a8a', players: ['Jose Luis Chilavert', 'Roque Santa Cruz', 'Julio Cesar Romero'], star: 'Julio Enciso', dishes: ['Sopa Paraguaya', 'Chipa', 'Mbejú'] },
  { id: 'uruguay', name: 'Uruguay', flag: '🇺🇾', color1: '#38bdf8', color2: '#ffffff', players: ['Luis Suárez', 'Diego Forlán', 'Enzo Francescoli'], star: 'Federico Valverde', dishes: ['Chivito', 'Asado', 'Martín Fierro'] },
  { id: 'australia', name: 'Australia', flag: '🇦🇺', color1: '#1e3a8a', color2: '#eab308', players: ['Tim Cahill', 'Harry Kewell', 'Mark Viduka'], star: 'Nestory Irankunda', dishes: ['Meat Pie', 'Vegemite Toast', 'Pavlova'] },
  { id: 'iran', name: 'Iran', flag: '🇮🇷', color1: '#16a34a', color2: '#dc2626', players: ['Ali Daei', 'Ali Karimi', 'Mehdi Mahdavikia'], star: 'Mehdi Taremi', dishes: ['Chelo Kebab', 'Ghormeh Sabzi', 'Fesenjan'] },
  { id: 'iraq', name: 'Iraq', flag: '🇮🇶', color1: '#ffffff', color2: '#16a34a', players: ['Younis Mahmoud', 'Ahmed Radhi', 'Nashat Akram'], star: 'Aymen Hussein', dishes: ['Masgouf', 'Biryani', 'Kleicha'] },
  { id: 'japan', name: 'Japan', flag: '🇯🇵', color1: '#1e3a8a', color2: '#ffffff', players: ['Hidetoshi Nakata', 'Shunsuke Nakamura', 'Keisuke Honda'], star: 'Kaoru Mitoma', dishes: ['Sushi', 'Ramen', 'Tempura'] },
  { id: 'jordan', name: 'Jordan', flag: '🇯🇴', color1: '#dc2626', color2: '#ffffff', players: ['Amer Deeb', 'Baha Abdel-Rahman', 'Odai Al-Saify'], star: 'Mousa Al-Tamari', dishes: ['Mansaf', 'Falafel', 'Kanafeh'] },
  { id: 'qatar', name: 'Qatar', flag: '🇶🇦', color1: '#881337', color2: '#ffffff', players: ['Mansour Muftah', 'Sebastián Soria', 'Hassan Al-Haydos'], star: 'Akram Afif', dishes: ['Machboos', 'Luqaimat', 'Harees'] },
  { id: 'saudi-arabia', name: 'Saudi Arabia', flag: '🇸🇦', color1: '#16a34a', color2: '#ffffff', players: ['Majed Abdullah', 'Sami Al-Jaber', 'Saeed Al-Owairan'], star: 'Salem Al-Dawsari', dishes: ['Kabsa', 'Jareesh', 'Mutabbaq'] },
  { id: 'south-korea', name: 'South Korea', flag: '🇰🇷', color1: '#dc2626', color2: '#1e3a8a', players: ['Park Ji-sung', 'Cha Bum-kun', 'Ahn Jung-hwan'], star: 'Son Heung-min', dishes: ['Kimchi', 'Bulgogi', 'Bibimbap'] },
  { id: 'uzbekistan', name: 'Uzbekistan', flag: '🇺🇿', color1: '#0ea5e9', color2: '#ffffff', players: ['Maksim Shatskikh', 'Server Djeparov', 'Odil Ahmedov'], star: 'Eldor Shomurodov', dishes: ['Plov', 'Somsa', 'Lagman'] },
  { id: 'algeria', name: 'Algeria', flag: '🇩🇿', color1: '#16a34a', color2: '#ffffff', players: ['Rabah Madjer', 'Lakhdar Belloumi', 'Rachid Mekhloufi'], star: 'Riyad Mahrez', dishes: ['Couscous', 'Shakshouka', 'Tajine'] },
  { id: 'caboverde', name: 'Cabo Verde', flag: '🇨🇻', color1: '#1e3a8a', color2: '#dc2626', players: ['Ryan Mendes', 'Heldon Ramos', 'Babanco'], star: 'Ryan Mendes', dishes: ['Cachupa', 'Pastel', 'Pudim de Leite'] },
  { id: 'cote-divoire', name: 'Côte d’Ivoire', flag: '🇨🇮', color1: '#f97316', color2: '#16a34a', players: ['Didier Drogba', 'Yaya Touré', 'Laurent Pokou'], star: 'Sébastien Haller', dishes: ['Garba', 'Aloko', 'Kedjenou'] },
  { id: 'dr-congo', name: 'DR Congo', flag: '🇨🇩', color1: '#0ea5e9', color2: '#dc2626', players: ['Shabani Nonda', 'Dieumerci Mbokani', 'Robert Kidiaba'], star: 'Chancel Mbemba', dishes: ['Moambé Chicken', 'Fufu', 'Chikwangue'] },
  { id: 'egypt', name: 'Egypt', flag: '🇪🇬', color1: '#dc2626', color2: '#ffffff', players: ['Mohamed Aboutrika', 'Hossam Hassan', 'Essam El-Hadary'], star: 'Mohamed Salah', dishes: ['Koshary', 'Ful Medames', 'Mulukhiyah'] },
  { id: 'ghana', name: 'Ghana', flag: '🇬🇭', color1: '#dc2626', color2: '#eab308', players: ['Abedi Pele', 'Asamoah Gyan', 'Tony Yeboah'], star: 'Mohammed Kudus', dishes: ['Jollof Rice', 'Fufu', 'Kelewele'] },
  { id: 'morocco', name: 'Morocco', flag: '🇲🇦', color1: '#dc2626', color2: '#16a34a', players: ['Mustapha Hadji', 'Noureddine Naybet', 'Larbi Benbarek'], star: 'Achraf Hakimi', dishes: ['Tagine', 'Couscous', 'Harira'] },
  { id: 'senegal', name: 'Senegal', flag: '🇸🇳', color1: '#16a34a', color2: '#eab308', players: ['Sadio Mané', 'El Hadji Diouf', 'Henri Camara'], star: 'Sadio Mané', dishes: ['Thiéboudienne', 'Yassa Poulet', 'Maafe'] },
  { id: 'south-africa', name: 'South Africa', flag: '🇿🇦', color1: '#16a34a', color2: '#eab308', players: ['Benni McCarthy', 'Lucas Radebe', 'Doctor Khumalo'], star: 'Percy Tau', dishes: ['Biltong', 'Bobotie', 'Bunny Chow'] },
  { id: 'tunisia', name: 'Tunisia', flag: '🇹🇳', color1: '#dc2626', color2: '#ffffff', players: ['Radhi Jaïdi', 'Wahbi Khazri', 'Tarek Dhiab'], star: 'Ellyes Skhiri', dishes: ['Couscous', 'Brik', 'Lablabi'] },
  { id: 'curacao', name: 'Curaçao', flag: '🇨🇼', color1: '#1e3a8a', color2: '#eab308', players: ['Cuco Martina', 'Leandro Bacuna', 'Charlison Benschop'], star: 'Juninho Bacuna', dishes: ['Keshi Yena', 'Stobá', 'Sopito'] },
  { id: 'haiti', name: 'Haiti', flag: '🇭🇹', color1: '#1e3a8a', color2: '#dc2626', players: ['Emmanuel Sanon', 'Wagneau Eloi', 'Johnny Placide'], star: 'Frantzdy Pierrot', dishes: ['Griot', 'Soup Joumou', 'Akasan'] },
  { id: 'panama', name: 'Panama', flag: '🇵🇦', color1: '#dc2626', color2: '#1e3a8a', players: ['Julio Dely Valdés', 'Blas Pérez', 'Luis Tejada'], star: 'Adalberto Carrasquilla', dishes: ['Sancocho', 'Ropa Vieja', 'Carimañolas'] },
  { id: 'new-zealand', name: 'New Zealand', flag: '🇳🇿', color1: '#ffffff', color2: '#111827', players: ['Wynton Rufer', 'Ryan Nelsen', 'Ivan Vicelich'], star: 'Chris Wood', dishes: ['Hāngī', 'Pavlova', 'Whitebait Fritter'] }
];

const publicDir = path.join(process.cwd(), 'public');
const assetsDir = path.join(publicDir, 'assets');
const playersDir = path.join(assetsDir, 'players');
const dishesDir = path.join(assetsDir, 'dishes');

if (!fs.existsSync(assetsDir)) fs.mkdirSync(assetsDir, { recursive: true });
if (!fs.existsSync(playersDir)) fs.mkdirSync(playersDir, { recursive: true });
if (!fs.existsSync(dishesDir)) fs.mkdirSync(dishesDir, { recursive: true });

function getPlayerCardSVG(name, type, flag, color1, color2) {
  const isStar = type === 'star';
  const role = isStar ? 'STAR PLAYER' : 'LEGEND';
  const cardGradStart = isStar ? '#f59e0b' : '#cbd5e1';
  const cardGradEnd = isStar ? '#0f172a' : '#1e293b';
  const borderColor = isStar ? '#fbbf24' : '#94a3b8';
  const safeName = name.replace(/'/g, "\\'");
  
  return `<svg xmlns="http://www.w3.org/2000/svg" width="200" height="280" viewBox="0 0 200 280">
  <defs>
    <linearGradient id="cardGrad-${name.replace(/[^a-zA-Z0-9]/g, '')}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="${cardGradStart}" />
      <stop offset="100%" stop-color="${cardGradEnd}" />
    </linearGradient>
    <linearGradient id="jerseyGrad-${name.replace(/[^a-zA-Z0-9]/g, '')}" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="${color1}" />
      <stop offset="100%" stop-color="${color2}" />
    </linearGradient>
  </defs>
  
  <rect x="5" y="5" width="190" height="270" rx="16" fill="url(#cardGrad-${name.replace(/[^a-zA-Z0-9]/g, '')})" stroke="${borderColor}" stroke-width="3" />
  
  <path d="M 5,60 L 195,60 M 5,210 L 195,210" stroke="${borderColor}" stroke-width="1" opacity="0.2" />
  <circle cx="100" cy="130" r="45" fill="none" stroke="${borderColor}" stroke-width="1" opacity="0.15" />
  
  <text x="25" y="32" font-family="'Outfit', sans-serif" font-size="18" text-anchor="middle">${flag}</text>
  <rect x="135" y="18" width="45" height="18" rx="4" fill="${isStar ? '#fbbf24' : '#94a3b8'}" />
  <text x="157.5" y="30" font-family="'Outfit', sans-serif" font-size="8" font-weight="800" fill="#0f172a" text-anchor="middle">${role}</text>
  
  <g transform="translate(0, 20)">
    <circle cx="100" cy="90" r="14" fill="#ffffff" fill-opacity="0.9" />
    <path d="M 85,110 Q 100,105 115,110 L 112,160 L 88,160 Z" fill="url(#jerseyGrad-${name.replace(/[^a-zA-Z0-9]/g, '')})" />
    <path d="M 88,160 L 80,205 L 94,208 L 98,160" fill="${color1}" />
    <path d="M 112,160 L 120,205 L 106,208 L 102,160" fill="${color1}" />
    <path d="M 85,110 L 70,135 L 78,140 L 90,118" fill="url(#jerseyGrad-${name.replace(/[^a-zA-Z0-9]/g, '')})" />
    <path d="M 115,110 L 130,135 L 122,140 L 110,118" fill="url(#jerseyGrad-${name.replace(/[^a-zA-Z0-9]/g, '')})" />
    <circle cx="135" cy="205" r="7" fill="#ffffff" stroke="#000000" stroke-width="1" />
  </g>
  
  <rect x="15" y="220" width="170" height="45" rx="8" fill="#000000" fill-opacity="0.5" stroke="${borderColor}" stroke-opacity="0.3" />
  <text x="100" y="240" font-family="'Outfit', sans-serif" font-size="11" font-weight="800" fill="#ffffff" text-anchor="middle">${safeName}</text>
  <text x="100" y="254" font-family="'Outfit', sans-serif" font-size="8" font-weight="600" fill="${isStar ? '#fbbf24' : '#cbd5e1'}" text-anchor="middle" letter-spacing="1">${role}</text>
</svg>`;
}

function getDishCardSVG(name, flag) {
  const safeName = name.replace(/'/g, "\\'");
  return `<svg xmlns="http://www.w3.org/2000/svg" width="300" height="200" viewBox="0 0 300 200">
  <defs>
    <linearGradient id="dishGrad-${name.replace(/[^a-zA-Z0-9]/g, '')}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#334155" />
    </linearGradient>
  </defs>
  
  <rect x="5" y="5" width="290" height="190" rx="16" fill="url(#dishGrad-${name.replace(/[^a-zA-Z0-9]/g, '')})" stroke="#475569" stroke-width="2" />
  
  <g transform="translate(150, 85)">
    <ellipse cx="0" cy="25" rx="65" ry="12" fill="#1e293b" opacity="0.6" />
    <ellipse cx="0" cy="20" rx="60" ry="15" fill="#e2e8f0" stroke="#cbd5e1" stroke-width="2" />
    <path d="M -50,15 C -30,45 30,45 50,15 Z" fill="#f59e0b" opacity="0.85" />
    <circle cx="-15" cy="18" r="8" fill="#10b981" />
    <circle cx="15" cy="22" r="7" fill="#ef4444" />
    <circle cx="0" cy="12" r="10" fill="#f59e0b" />
    <path d="M -40,15 C -40,-25 40,-25 40,15 Z" fill="none" stroke="#f1f5f9" stroke-width="2" opacity="0.15" />
    <path d="M -20,-10 Q -15,-20 -20,-30 T -20,-45" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" opacity="0.4" />
    <path d="M 0,-10 Q 5,-20 0,-30 T 0,-45" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" opacity="0.6" />
    <path d="M 20,-10 Q 25,-20 20,-30 T 20,-45" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" opacity="0.4" />
  </g>
  
  <rect x="15" y="145" width="270" height="40" rx="8" fill="#000000" fill-opacity="0.6" stroke="#475569" stroke-opacity="0.5" />
  <text x="150" y="169" font-family="'Outfit', sans-serif" font-size="13" font-weight="700" fill="#ffffff" text-anchor="middle">${safeName}</text>
  <text x="35" y="170" font-family="'Outfit', sans-serif" font-size="16" text-anchor="middle">${flag}</text>
</svg>`;
}

for (const team of teams) {
  const starSVG = getPlayerCardSVG(team.star, 'star', team.flag, team.color1, team.color2);
  fs.writeFileSync(path.join(playersDir, `${team.id}-star.svg`), starSVG);
  
  for (let i = 0; i < team.players.length; i++) {
    const legendName = team.players[i];
    const legendSVG = getPlayerCardSVG(legendName, 'legend', team.flag, team.color1, team.color2);
    fs.writeFileSync(path.join(playersDir, `${team.id}-legend-${i}.svg`), legendSVG);
  }
  
  for (let i = 0; i < team.dishes.length; i++) {
    const dishName = team.dishes[i];
    const dishSVG = getDishCardSVG(dishName, team.flag);
    fs.writeFileSync(path.join(dishesDir, `${team.id}-dish-${i}.svg`), dishSVG);
  }
}
