const teamsData = [
  { id: 'canada', name: 'Canada', flag: '🇨🇦', titles: 0, players: ['Atiba Hutchinson', 'Craig Forrest', 'Dwayne De Rosario'], group: 'Group A', star: 'Alphonso Davies', coach: 'Jesse Marsch', chance: 3 },
  { id: 'mexico', name: 'Mexico', flag: '🇲🇽', titles: 0, players: ['Hugo Sánchez', 'Rafael Márquez', 'Javier Hernández'], group: 'Group B', star: 'Santiago Giménez', coach: 'Javier Aguirre', chance: 4 },
  { id: 'usa', name: 'USA', flag: '🇺🇸', titles: 0, players: ['Landon Donovan', 'Clint Dempsey', 'Cobi Jones'], group: 'Group A', star: 'Christian Pulisic', coach: 'Mauricio Pochettino', chance: 5 },
  { id: 'austria', name: 'Austria', flag: '🇦🇹', titles: 0, players: ['David Alaba', 'Toni Polster', 'Hans Krankl'], group: 'Group H', star: 'Konrad Laimer', coach: 'Ralf Rangnick', chance: 2 },
  { id: 'belgium', name: 'Belgium', flag: '🇧🇪', titles: 0, players: ['Eden Hazard', 'Vincent Kompany', 'Paul Van Himst'], group: 'Group G', star: 'Kevin De Bruyne', coach: 'Domenico Tedesco', chance: 7 },
  { id: 'bosnia', name: 'Bosnia and Herzegovina', flag: '🇧🇦', titles: 0, players: ['Edin Džeko', 'Miralem Pjanić', 'Sergej Barbarez'], group: 'Group L', star: 'Edin Džeko', coach: 'Sergej Barbarez', chance: 1 },
  { id: 'croatia', name: 'Croatia', flag: '🇭🇷', titles: 0, players: ['Luka Modrić', 'Davor Šuker', 'Zvonimir Boban'], group: 'Group K', star: 'Luka Modrić', coach: 'Zlatko Dalić', chance: 5 },
  { id: 'czechia', name: 'Czechia', flag: '🇨🇿', titles: 0, players: ['Pavel Nedvěd', 'Petr Čech', 'Josef Masopust'], group: 'Group J', star: 'Patrik Schick', coach: 'Ivan Hašek', chance: 2 },
  { id: 'england', name: 'England', flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', titles: 1, players: ['Bobby Charlton', 'Gary Lineker', 'Bobby Moore'], group: 'Group F', star: 'Jude Bellingham', coach: 'Thomas Tuchel', chance: 11 },
  { id: 'france', name: 'France', flag: '🇫🇷', titles: 2, players: ['Zinedine Zidane', 'Michel Platini', 'Thierry Henry'], group: 'Group D', star: 'Kylian Mbappé', coach: 'Didier Deschamps', chance: 16 },
  { id: 'germany', name: 'Germany', flag: '🇩🇪', titles: 4, players: ['Franz Beckenbauer', 'Gerd Müller', 'Miroslav Klose'], group: 'Group C', star: 'Florian Wirtz', coach: 'Julian Nagelsmann', chance: 12 },
  { id: 'netherlands', name: 'Netherlands', flag: '🇳🇱', titles: 0, players: ['Johan Cruyff', 'Marco van Basten', 'Ruud Gullit'], group: 'Group J', star: 'Virgil van Dijk', coach: 'Ronald Koeman', chance: 8 },
  { id: 'norway', name: 'Norway', flag: '🇳🇴', titles: 0, players: ['Erik Thorstvedt', 'John Carew', 'Tore André Flo'], group: 'Group F', star: 'Erling Haaland', coach: 'Ståle Solbakken', chance: 6 },
  { id: 'portugal', name: 'Portugal', flag: '🇵🇹', titles: 0, players: ['Cristiano Ronaldo', 'Eusébio', 'Luís Figo'], group: 'Group I', star: 'Bruno Fernandes', coach: 'Roberto Martínez', chance: 9 },
  { id: 'scotland', name: 'Scotland', flag: '🏴󠁧󠁢󠁳󠁣󠁴󠁿', titles: 0, players: ['Kenny Dalglish', 'Denis Law', 'Graeme Souness'], group: 'Group I', star: 'Andrew Robertson', coach: 'Steve Clarke', chance: 1 },
  { id: 'spain', name: 'Spain', flag: '🇪🇸', titles: 1, players: ['Andres Iniesta', 'Xavi Hernandez', 'Iker Casillas'], group: 'Group H', star: 'Lamine Yamal', coach: 'Luis de la Fuente', chance: 13 },
  { id: 'sweden', name: 'Sweden', flag: '🇸🇪', titles: 0, players: ['Zlatan Ibrahimović', 'Henrik Larsson', 'Gunnar Nordahl'], group: 'Group D', star: 'Alexander Isak', coach: 'Jon Dahl Tomasson', chance: 3 },
  { id: 'switzerland', name: 'Switzerland', flag: '🇨🇭', titles: 0, players: ['Stephane Chapuisat', 'Alexander Frei', 'Xherdan Shaqiri'], group: 'Group L', star: 'Granit Xhaka', coach: 'Murat Yakin', chance: 4 },
  { id: 'turkiye', name: 'Türkiye', flag: '🇹🇷', titles: 0, players: ['Hakan Şükür', 'Rüştü Reçber', 'Tugay Kerimoğlu'], group: 'Group K', star: 'Hakan Çalhanoğlu', coach: 'Vincenzo Montella', chance: 3 },
  { id: 'argentina', name: 'Argentina', flag: '🇦🇷', titles: 3, players: ['Diego Maradona', 'Lionel Messi', 'Mario Kempes'], group: 'Group B', star: 'Lionel Messi', coach: 'Lionel Scaloni', chance: 14 },
  { id: 'brazil', name: 'Brazil', flag: '🇧🇷', titles: 5, players: ['Pelé', 'Ronaldo', 'Ronaldinho'], group: 'Group A', star: 'Vinícius Júnior', coach: 'Carlo Ancelotti', chance: 15 },
  { id: 'colombia', name: 'Colombia', flag: '🇨🇴', titles: 0, players: ['Carlos Valderrama', 'Radamel Falcao', 'Faustino Asprilla'], group: 'Group E', star: 'Luis Díaz', coach: 'Néstor Lorenzo', chance: 7 },
  { id: 'ecuador', name: 'Ecuador', flag: '🇪🇨', titles: 0, players: ['Alex Aguinaga', 'Antonio Valencia', 'Enner Valencia'], group: 'Group B', star: 'Moisés Caicedo', coach: 'Sebastián Beccacece', chance: 4 },
  { id: 'paraguay', name: 'Paraguay', flag: '🇵🇾', titles: 0, players: ['Jose Luis Chilavert', 'Roque Santa Cruz', 'Julio Cesar Romero'], group: 'Group D', star: 'Julio Enciso', coach: 'Gustavo Alfaro', chance: 3 },
  { id: 'uruguay', name: 'Uruguay', flag: '🇺🇾', titles: 2, players: ['Luis Suárez', 'Diego Forlán', 'Enzo Francescoli'], group: 'Group E', star: 'Federico Valverde', coach: 'Marcelo Bielsa', chance: 8 },
  { id: 'australia', name: 'Australia', flag: '🇦🇺', titles: 0, players: ['Tim Cahill', 'Harry Kewell', 'Mark Viduka'], group: 'Group G', star: 'Nestory Irankunda', coach: 'Tony Popovic', chance: 2 },
  { id: 'iran', name: 'Iran', flag: '🇮🇷', titles: 0, players: ['Ali Daei', 'Ali Karimi', 'Mehdi Mahdavikia'], group: 'Group I', star: 'Mehdi Taremi', coach: 'Amir Ghalenoei', chance: 2 },
  { id: 'iraq', name: 'Iraq', flag: '🇮🇶', titles: 0, players: ['Younis Mahmoud', 'Ahmed Radhi', 'Nashat Akram'], group: 'Group J', star: 'Aymen Hussein', coach: 'Jesús Casas', chance: 1 },
  { id: 'japan', name: 'Japan', flag: '🇯🇵', titles: 0, players: ['Hidetoshi Nakata', 'Shunsuke Nakamura', 'Keisuke Honda'], group: 'Group E', star: 'Kaoru Mitoma', coach: 'Hajime Moriyasu', chance: 5 },
  { id: 'jordan', name: 'Jordan', flag: '🇯🇴', titles: 0, players: ['Amer Deeb', 'Baha Abdel-Rahman', 'Odai Al-Saify'], group: 'Group K', star: 'Mousa Al-Tamari', coach: 'Jamal Sellami', chance: 1 },
  { id: 'qatar', name: 'Qatar', flag: '🇶🇦', titles: 0, players: ['Mansour Muftah', 'Sebastián Soria', 'Hassan Al-Haydos'], group: 'Group G', star: 'Akram Afif', coach: 'Tintín Márquez', chance: 1 },
  { id: 'saudi-arabia', name: 'Saudi Arabia', flag: '🇸🇦', titles: 0, players: ['Majed Abdullah', 'Sami Al-Jaber', 'Saeed Al-Owairan'], group: 'Group H', star: 'Salem Al-Dawsari', coach: 'Roberto Mancini', chance: 1 },
  { id: 'south-korea', name: 'South Korea', flag: '🇰🇷', titles: 0, players: ['Park Ji-sung', 'Cha Bum-kun', 'Ahn Jung-hwan'], group: 'Group F', star: 'Son Heung-min', coach: 'Hong Myung-bo', chance: 3 },
  { id: 'uzbekistan', name: 'Uzbekistan', flag: '🇺🇿', titles: 0, players: ['Maksim Shatskikh', 'Server Djeparov', 'Odil Ahmedov'], group: 'Group L', star: 'Eldor Shomurodov', coach: 'Srečko Katanec', chance: 1 },
  { id: 'algeria', name: 'Algeria', flag: '🇩🇿', titles: 0, players: ['Rabah Madjer', 'Lakhdar Belloumi', 'Rachid Mekhloufi'], group: 'Group L', star: 'Riyad Mahrez', coach: 'Vladimir Petković', chance: 3 },
  { id: 'caboverde', name: 'Cabo Verde', flag: '🇨🇻', titles: 0, players: ['Ryan Mendes', 'Heldon Ramos', 'Babanco'], group: 'Group B', star: 'Ryan Mendes', coach: 'Bubista', chance: 1 },
  { id: 'cote-divoire', name: 'Côte d’Ivoire', flag: '🇨🇮', titles: 0, players: ['Didier Drogba', 'Yaya Touré', 'Laurent Pokou'], group: 'Group F', star: 'Sébastien Haller', coach: 'Emerse Faé', chance: 4 },
  { id: 'dr-congo', name: 'DR Congo', flag: '🇨🇩', titles: 0, players: ['Shabani Nonda', 'Dieumerci Mbokani', 'Robert Kidiaba'], group: 'Group F', star: 'Chancel Mbemba', coach: 'Sébastien Desabre', chance: 2 },
  { id: 'egypt', name: 'Egypt', flag: '🇪🇬', titles: 0, players: ['Mohamed Aboutrika', 'Hossam Hassan', 'Essam El-Hadary'], group: 'Group G', star: 'Mohamed Salah', coach: 'Hossam Hassan', chance: 5 },
  { id: 'ghana', name: 'Ghana', flag: '🇬🇭', titles: 0, players: ['Abedi Pele', 'Asamoah Gyan', 'Tony Yeboah'], group: 'Group E', star: 'Mohammed Kudus', coach: 'Otto Addo', chance: 3 },
  { id: 'morocco', name: 'Morocco', flag: '🇲🇦', titles: 0, players: ['Mustapha Hadji', 'Noureddine Naybet', 'Larbi Benbarek'], group: 'Group C', star: 'Achraf Hakimi', coach: 'Walid Regragui', chance: 6 },
  { id: 'senegal', name: 'Senegal', flag: '🇸🇳', titles: 0, players: ['Sadio Mané', 'El Hadji Diouf', 'Henri Camara'], group: 'Group J', star: 'Sadio Mané', coach: 'Pape Thiaw', chance: 5 },
  { id: 'south-africa', name: 'South Africa', flag: '🇿🇦', titles: 0, players: ['Benni McCarthy', 'Lucas Radebe', 'Doctor Khumalo'], group: 'Group A', star: 'Percy Tau', coach: 'Hugo Broos', chance: 2 },
  { id: 'tunisia', name: 'Tunisia', flag: '🇹🇳', titles: 0, players: ['Radhi Jaïdi', 'Wahbi Khazri', 'Tarek Dhiab'], group: 'Group I', star: 'Ellyes Skhiri', coach: 'Faouzi Benzarti', chance: 2 },
  { id: 'curacao', name: 'Curaçao', flag: '🇨🇼', titles: 0, players: ['Cuco Martina', 'Leandro Bacuna', 'Charlison Benschop'], group: 'Group C', star: 'Juninho Bacuna', coach: 'Dick Advocaat', chance: 1 },
  { id: 'haiti', name: 'Haiti', flag: '🇭🇹', titles: 0, players: ['Emmanuel Sanon', 'Wagneau Eloi', 'Johnny Placide'], group: 'Group D', star: 'Frantzdy Pierrot', coach: 'Sébastien Migné', chance: 1 },
  { id: 'panama', name: 'Panama', flag: '🇵🇦', titles: 0, players: ['Julio Dely Valdés', 'Blas Pérez', 'Luis Tejada'], group: 'Group C', star: 'Adalberto Carrasquilla', coach: 'Thomas Christiansen', chance: 2 },
  { id: 'new-zealand', name: 'New Zealand', flag: '🇳🇿', titles: 0, players: ['Wynton Rufer', 'Ryan Nelsen', 'Ivan Vicelich'], group: 'Group K', star: 'Chris Wood', coach: 'Darren Bazeley', chance: 1 }
];

const tabButtons = document.querySelectorAll('.tab-btn');
const tabPanels = document.querySelectorAll('.tab-panel');

tabButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    tabButtons.forEach(b => b.classList.remove('active'));
    tabPanels.forEach(p => p.classList.remove('active'));
    
    btn.classList.add('active');
    const panelId = btn.id.replace('-btn', '').replace('tab-', 'panel-');
    document.getElementById(panelId).classList.add('active');

    if (panelId === 'panel-bracket') {
      loadBracketData();
    } else if (panelId === 'panel-prediction') {
      renderWinningChances();
    }
  });
});

const teamSearch = document.getElementById('team-search');
const teamsListContainer = document.getElementById('teams-list-container');
const teamDetailsContainer = document.getElementById('team-details-container');

const teamMetadata = {
  canada: { color1: '#ef4444', color2: '#ffffff', nickname: 'Les Rouges', years: '', runnerUp: 0, stats: { appearances: 3, finals: 0, semifinals: 0, wins: 0 } },
  mexico: { color1: '#15803d', color2: '#ef4444', nickname: 'El Tri', years: '', runnerUp: 0, stats: { appearances: 17, finals: 0, semifinals: 0, wins: 16 } },
  usa: { color1: '#1e3a8a', color2: '#ef4444', nickname: 'The Stars & Stripes', years: '', runnerUp: 0, stats: { appearances: 11, finals: 0, semifinals: 1, wins: 8 } },
  austria: { color1: '#dc2626', color2: '#ffffff', nickname: 'Das Team', years: '', runnerUp: 0, stats: { appearances: 7, finals: 0, semifinals: 1, wins: 12 } },
  belgium: { color1: '#facc15', color2: '#dc2626', nickname: 'The Red Devils', years: '', runnerUp: 0, stats: { appearances: 14, finals: 0, semifinals: 2, wins: 20 } },
  bosnia: { color1: '#1d4ed8', color2: '#facc15', nickname: 'Zmajevi', years: '', runnerUp: 0, stats: { appearances: 1, finals: 0, semifinals: 0, wins: 1 } },
  croatia: { color1: '#dc2626', color2: '#ffffff', nickname: 'Vatreni', years: '', runnerUp: 1, stats: { appearances: 6, finals: 1, semifinals: 3, wins: 13 } },
  czechia: { color1: '#1e3a8a', color2: '#dc2626', nickname: 'Naši', years: '', runnerUp: 2, stats: { appearances: 9, finals: 2, semifinals: 2, wins: 12 } },
  england: { color1: '#ffffff', color2: '#dc2626', nickname: 'The Three Lions', years: '1966', runnerUp: 0, stats: { appearances: 16, finals: 1, semifinals: 3, wins: 32 } },
  france: { color1: '#1d4ed8', color2: '#dc2626', nickname: 'Les Bleus', years: '1998, 2018', runnerUp: 2, stats: { appearances: 16, finals: 4, semifinals: 6, wins: 39 } },
  germany: { color1: '#111827', color2: '#dc2626', nickname: 'Nationalelf', years: '1954, 1974, 1990, 2014', runnerUp: 4, stats: { appearances: 20, finals: 8, semifinals: 13, wins: 68 } },
  netherlands: { color1: '#f97316', color2: '#ffffff', nickname: 'Oranje', years: '', runnerUp: 3, stats: { appearances: 11, finals: 3, semifinals: 5, wins: 30 } },
  norway: { color1: '#dc2626', color2: '#1d4ed8', nickname: 'Løvene', years: '', runnerUp: 0, stats: { appearances: 3, finals: 0, semifinals: 0, wins: 2 } },
  portugal: { color1: '#16a34a', color2: '#dc2626', nickname: 'Seleção das Quinas', years: '', runnerUp: 0, stats: { appearances: 8, finals: 0, semifinals: 2, wins: 15 } },
  scotland: { color1: '#1d4ed8', color2: '#ffffff', nickname: 'The Tartan Army', years: '', runnerUp: 0, stats: { appearances: 8, finals: 0, semifinals: 0, wins: 4 } },
  spain: { color1: '#dc2626', color2: '#eab308', nickname: 'La Roja', years: '2010', runnerUp: 0, stats: { appearances: 16, finals: 1, semifinals: 2, wins: 31 } },
  sweden: { color1: '#1d4ed8', color2: '#eab308', nickname: 'Blågult', years: '', runnerUp: 1, stats: { appearances: 12, finals: 1, semifinals: 4, wins: 19 } },
  switzerland: { color1: '#dc2626', color2: '#ffffff', nickname: 'Nati', years: '', runnerUp: 0, stats: { appearances: 12, finals: 0, semifinals: 0, wins: 12 } },
  turkiye: { color1: '#dc2626', color2: '#ffffff', nickname: 'Bizim Çocuklar', years: '', runnerUp: 0, stats: { appearances: 2, finals: 0, semifinals: 1, wins: 5 } },
  argentina: { color1: '#38bdf8', color2: '#ffffff', nickname: 'La Albiceleste', years: '1978, 1986, 2022', runnerUp: 3, stats: { appearances: 18, finals: 6, semifinals: 6, wins: 47 } },
  brazil: { color1: '#eab308', color2: '#16a34a', nickname: 'Canarinha', years: '1958, 1962, 1970, 1994, 2002', runnerUp: 2, stats: { appearances: 22, finals: 7, semifinals: 11, wins: 76 } },
  colombia: { color1: '#eab308', color2: '#1d4ed8', nickname: 'Los Cafeteros', years: '', runnerUp: 0, stats: { appearances: 6, finals: 0, semifinals: 0, wins: 9 } },
  ecuador: { color1: '#eab308', color2: '#1d4ed8', nickname: 'La Tri', years: '', runnerUp: 0, stats: { appearances: 4, finals: 0, semifinals: 0, wins: 5 } },
  paraguay: { color1: '#dc2626', color2: '#1e3a8a', nickname: 'La Albirroja', years: '', runnerUp: 0, stats: { appearances: 8, finals: 0, semifinals: 0, wins: 7 } },
  uruguay: { color1: '#38bdf8', color2: '#ffffff', nickname: 'La Celeste', years: '1930, 1950', runnerUp: 0, stats: { appearances: 14, finals: 2, semifinals: 5, wins: 24 } },
  australia: { color1: '#1e3a8a', color2: '#eab308', nickname: 'Socceroos', years: '', runnerUp: 0, stats: { appearances: 6, finals: 0, semifinals: 0, wins: 4 } },
  iran: { color1: '#16a34a', color2: '#dc2626', nickname: 'Team Melli', years: '', runnerUp: 0, stats: { appearances: 6, finals: 0, semifinals: 0, wins: 3 } },
  iraq: { color1: '#ffffff', color2: '#16a34a', nickname: 'Lions of Mesopotamia', years: '', runnerUp: 0, stats: { appearances: 1, finals: 0, semifinals: 0, wins: 0 } },
  japan: { color1: '#1e3a8a', color2: '#ffffff', nickname: 'Samurai Blue', years: '', runnerUp: 0, stats: { appearances: 7, finals: 0, semifinals: 0, wins: 7 } },
  jordan: { color1: '#dc2626', color2: '#ffffff', nickname: 'The Chivalrous', years: '', runnerUp: 0, stats: { appearances: 0, finals: 0, semifinals: 0, wins: 0 } },
  qatar: { color1: '#881337', color2: '#ffffff', nickname: 'The Maroon', years: '', runnerUp: 0, stats: { appearances: 1, finals: 0, semifinals: 0, wins: 0 } },
  'saudi-arabia': { color1: '#16a34a', color2: '#ffffff', nickname: 'Green Falcons', years: '', runnerUp: 0, stats: { appearances: 6, finals: 0, semifinals: 0, wins: 4 } },
  'south-korea': { color1: '#dc2626', color2: '#1e3a8a', nickname: 'Taegeuk Warriors', years: '', runnerUp: 0, stats: { appearances: 11, finals: 0, semifinals: 1, wins: 7 } },
  uzbekistan: { color1: '#0ea5e9', color2: '#ffffff', nickname: 'White Wolves', years: '', runnerUp: 0, stats: { appearances: 0, finals: 0, semifinals: 0, wins: 0 } },
  algeria: { color1: '#16a34a', color2: '#ffffff', nickname: 'Les Fennecs', years: '', runnerUp: 0, stats: { appearances: 4, finals: 0, semifinals: 0, wins: 3 } },
  caboverde: { color1: '#1e3a8a', color2: '#dc2626', nickname: 'Tubarões Azuis', years: '', runnerUp: 0, stats: { appearances: 0, finals: 0, semifinals: 0, wins: 0 } },
  'cote-divoire': { color1: '#f97316', color2: '#16a34a', nickname: 'Les Éléphants', years: '', runnerUp: 0, stats: { appearances: 3, finals: 0, semifinals: 0, wins: 3 } },
  'dr-congo': { color1: '#0ea5e9', color2: '#dc2626', nickname: 'Les Léopards', years: '', runnerUp: 0, stats: { appearances: 1, finals: 0, semifinals: 0, wins: 0 } },
  egypt: { color1: '#dc2626', color2: '#ffffff', nickname: 'The Pharaohs', years: '', runnerUp: 0, stats: { appearances: 3, finals: 0, semifinals: 0, wins: 0 } },
  ghana: { color1: '#dc2626', color2: '#eab308', nickname: 'Black Stars', years: '', runnerUp: 0, stats: { appearances: 4, finals: 0, semifinals: 0, wins: 5 } },
  morocco: { color1: '#dc2626', color2: '#16a34a', nickname: 'Atlas Lions', years: '', runnerUp: 0, stats: { appearances: 6, finals: 0, semifinals: 1, wins: 5 } },
  senegal: { color1: '#16a34a', color2: '#eab308', nickname: 'Lions of Teranga', years: '', runnerUp: 0, stats: { appearances: 3, finals: 0, semifinals: 0, wins: 5 } },
  'south-africa': { color1: '#16a34a', color2: '#eab308', nickname: 'Bafana Bafana', years: '', runnerUp: 0, stats: { appearances: 3, finals: 0, semifinals: 0, wins: 2 } },
  tunisia: { color1: '#dc2626', stop2: '#ffffff', nickname: 'Eagles of Carthage', years: '', runnerUp: 0, stats: { appearances: 6, finals: 0, semifinals: 0, wins: 3 } },
  curacao: { color1: '#1e3a8a', color2: '#eab308', nickname: 'La Familia Azul', years: '', runnerUp: 0, stats: { appearances: 0, finals: 0, semifinals: 0, wins: 0 } },
  haiti: { color1: '#1e3a8a', color2: '#dc2626', nickname: 'Les Grenadiers', years: '', runnerUp: 0, stats: { appearances: 1, finals: 0, semifinals: 0, wins: 0 } },
  panama: { color1: '#dc2626', color2: '#1e3a8a', nickname: 'Los Canaleros', years: '', runnerUp: 0, stats: { appearances: 1, finals: 0, semifinals: 0, wins: 0 } },
  'new-zealand': { color1: '#ffffff', color2: '#111827', nickname: 'All Whites', years: '', runnerUp: 0, stats: { appearances: 2, finals: 0, semifinals: 0, wins: 0 } }
};

teamsData.forEach(t => {
  const meta = teamMetadata[t.id] || { color1: '#0f172a', color2: '#1e293b', nickname: '', years: '', runnerUp: 0, stats: { appearances: 0, finals: 0, semifinals: 0, wins: 0 } };
  t.color1 = meta.color1;
  t.color2 = meta.color2 || meta.color1;
  t.nickname = meta.nickname;
  t.years = meta.years;
  t.runnerUp = meta.runnerUp;
  t.stats = meta.stats;
});

function getTextColor(hex) {
  if (!hex) return '#ffffff';
  const c = hex.substring(1);
  if (c === 'ffffff') return '#0f172a';
  const rgb = parseInt(c, 16);
  const r = (rgb >> 16) & 0xff;
  const g = (rgb >> 8) & 0xff;
  const b = (rgb >> 0) & 0xff;
  const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return luma > 180 ? '#0f172a' : '#ffffff';
}

function getSortedTeams() {
  return [...teamsData].sort((a, b) => b.titles - a.titles || a.name.localeCompare(b.name));
}

function renderTeams(filter = '') {
  teamsListContainer.innerHTML = '';
  const sorted = getSortedTeams();
  const filtered = sorted.filter(t => t.name.toLowerCase().includes(filter.toLowerCase()));

  filtered.forEach(team => {
    const item = document.createElement('div');
    item.className = 'team-item';
    item.innerHTML = `
      <div class="team-meta">
        <span class="team-flag">${team.flag}</span>
        <div class="team-info-block">
          <span class="team-name">${team.name}</span>
          <span class="team-titles-sub">${team.titles} ${team.titles === 1 ? 'Title' : 'Titles'}</span>
        </div>
      </div>
    `;
    item.addEventListener('click', () => {
      document.querySelectorAll('.team-item').forEach(i => i.classList.remove('active'));
      item.classList.add('active');
      showTeamDetails(team);
    });
    teamsListContainer.appendChild(item);
  });
}

const teamDishes = {
  canada: ['Poutine', 'Butter Tarts', 'Tourtière'],
  mexico: ['Tacos', 'Mole Poblano', 'Chiles en Nogada'],
  usa: ['Hamburger', 'Apple Pie', 'Clam Chowder'],
  austria: ['Wiener Schnitzel', 'Sachertorte', 'Apfelstrudel'],
  belgium: ['Moules-Frites', 'Belgian Waffles', 'Carbonnade Flamande'],
  bosnia: ['Ćevapi', 'Burek', 'Klepe'],
  croatia: ['Peka', 'Crni Rižot', 'Fritule'],
  czechia: ['Vepřo Knedlo Zelo', 'Svíčková', 'Trdelník'],
  england: ['Fish and Chips', 'Sunday Roast', 'Shepherd\'s Pie'],
  france: ['Coq au Vin', 'Ratatouille', 'Crème Brûlée'],
  germany: ['Bratwurst', 'Sauerkraut', 'Pretzel'],
  netherlands: ['Stroopwafel', 'Bitterballen', 'Stamppot'],
  norway: ['Fårikål', 'Gravlaks', 'Lutefisk'],
  portugal: ['Bacalhau à Brás', 'Pastel de Nata', 'Caldo Verde'],
  scotland: ['Haggis', 'Scotch Pie', 'Cranachan'],
  spain: ['Paella', 'Tortilla Española', 'Gazpacho'],
  sweden: ['Köttbullar', 'Gravlax', 'Smörgåstårta'],
  switzerland: ['Fondue', 'Raclette', 'Rösti'],
  turkiye: ['Kebab', 'Baklava', 'Pide'],
  argentina: ['Asado', 'Empanadas', 'Dulce de Leche'],
  brazil: ['Feijoada', 'Pão de Queijo', 'Brigadeiro'],
  colombia: ['Bandeja Paisa', 'Arepas', 'Ajiaco'],
  ecuador: ['Ceviche', 'Llapingachos', 'Locro de Papa'],
  paraguay: ['Sopa Paraguaya', 'Chipa', 'Mbejú'],
  uruguay: ['Chivito', 'Asado', 'Martín Fierro'],
  australia: ['Meat Pie', 'Vegemite Toast', 'Pavlova'],
  iran: ['Chelo Kebab', 'Ghormeh Sabzi', 'Fesenjan'],
  iraq: ['Masgouf', 'Biryani', 'Kleicha'],
  japan: ['Sushi', 'Ramen', 'Tempura'],
  jordan: ['Mansaf', 'Falafel', 'Kanafeh'],
  qatar: ['Machboos', 'Luqaimat', 'Harees'],
  'saudi-arabia': ['Kabsa', 'Jareesh', 'Mutabbaq'],
  'south-korea': ['Kimchi', 'Bulgogi', 'Bibimbap'],
  uzbekistan: ['Plov', 'Somsa', 'Lagman'],
  algeria: ['Couscous', 'Shakshouka', 'Tajine'],
  caboverde: ['Cachupa', 'Pastel', 'Pudim de Leite'],
  'cote-divoire': ['Garba', 'Aloko', 'Kedjenou'],
  'dr-congo': ['Moambé Chicken', 'Fufu', 'Chikwangue'],
  egypt: ['Koshary', 'Ful Medames', 'Mulukhiyah'],
  ghana: ['Jollof Rice', 'Fufu', 'Kelewele'],
  morocco: ['Tagine', 'Couscous', 'Harira'],
  senegal: ['Thiéboudienne', 'Yassa Poulet', 'Maafe'],
  'south-africa': ['Biltong', 'Bobotie', 'Bunny Chow'],
  tunisia: ['Couscous', 'Brik', 'Lablabi'],
  curacao: ['Keshi Yena', 'Stobá', 'Sopito'],
  haiti: ['Griot', 'Soup Joumou', 'Akasan'],
  panama: ['Sancocho', 'Ropa Vieja', 'Carimañolas'],
  'new-zealand': ['Hāngī', 'Pavlova', 'Whitebait Fritter']
};

function showTeamDetails(team) {
  const dishes = teamDishes[team.id] || [];
  const textColor = getTextColor(team.color1);
  const secondaryStyle = team.nickname ? ` | ${team.nickname}` : '';

  teamDetailsContainer.innerHTML = `
    <div class="team-details-header" style="background: linear-gradient(135deg, ${team.color1}, ${team.color2}); color: ${textColor};">
      <div class="details-identity">
        <span class="details-flag">${team.flag}</span>
        <div>
          <h2 style="color: inherit; margin: 0; font-size: 26px; font-weight: 800; letter-spacing: -0.5px;">${team.name.toUpperCase()}</h2>
          <span style="opacity: 0.85; font-size: 13px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase;">
            ${team.titles} ${team.titles === 1 ? 'Title' : 'Titles'}${secondaryStyle}
          </span>
        </div>
      </div>
      <div class="selected-team-badge">Selected Team</div>
    </div>

    <div class="historical-achievements-banner">
      <div class="achievements-title">Historical Achievements</div>
      <div class="achievements-content">
        <strong>${team.titles} ${team.titles === 1 ? 'Title' : 'Titles'}${team.years ? `: ${team.years}` : ''}</strong>
        <span class="divider">|</span>
        <span>Runner-up: ${team.runnerUp || 0}</span>
      </div>
    </div>

    <div class="details-grid">
      <div class="details-stats-row">
        <div class="details-section-card">
          <h3>World Cup 2026 Info</h3>
          <div class="stats-grid">
            <div class="stat-item">
              <span class="stat-label">Group Placement</span>
              <span class="stat-value">${team.group}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">Manager / Coach</span>
              <span class="stat-value">${team.coach}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">Star Player</span>
              <span class="stat-value">${team.star}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">Win Probability</span>
              <span class="stat-value">${team.chance}%</span>
            </div>
          </div>
        </div>
        
        <div class="details-section-card">
          <h3>All-Time Team Stats</h3>
          <div class="stats-grid">
            <div class="stat-item">
              <span class="stat-label">World Cups</span>
              <span class="stat-value">${team.stats.appearances} (2026)</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">Finals Played</span>
              <span class="stat-value">${team.stats.finals}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">Semi-Finals</span>
              <span class="stat-value">${team.stats.semifinals}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">All-Time Wins</span>
              <span class="stat-value">${team.stats.wins}</span>
            </div>
          </div>
        </div>
      </div>

      <div class="players-section">
        <h3>Squad Stars & Legends</h3>
        <div class="players-deck">
          <div class="player-card-visual">
            <img src="/assets/players/${team.id}-star.jpg" alt="${team.star}">
            <div class="player-name" style="margin-top: 8px; font-weight: 700; color: var(--text-dark);">${team.star}</div>
            <div class="player-desc" style="font-size: 12px; color: var(--text-muted);">Star Player</div>
          </div>
          ${team.players.map((p, idx) => `
            <div class="player-card-visual">
              <img src="/assets/players/${team.id}-legend-${idx}.jpg" alt="${p}">
              <div class="player-name" style="margin-top: 8px; font-weight: 700; color: var(--text-dark);">${p}</div>
              <div class="player-desc" style="font-size: 12px; color: var(--text-muted);">Legend</div>
            </div>
          `).join('')}
        </div>
      </div>

      <div class="dishes-section">
        <h3>Taste of the Nation (Popular Dishes)</h3>
        <div class="dishes-grid">
          ${dishes.map((dish, idx) => `
            <div class="dish-card-visual">
              <img src="/assets/dishes/${team.id}-dish-${idx}.jpg" alt="${dish}">
              <div class="dish-name-label">${dish}</div>
            </div>
          `).join('')}
        </div>
      </div>
    </div>
  `;
}

teamSearch.addEventListener('input', (e) => {
  renderTeams(e.target.value);
});

let currentBracket = null;

function loadBracketData() {
  fetch('http://localhost:3000/bracket.json')
    .then(res => res.json())
    .then(data => {
      currentBracket = data;
      renderBracket(data);
    });
}

function getTeamFlag(name) {
  if (!name) return '';
  const team = teamsData.find(t => t.name.toLowerCase() === name.toLowerCase());
  return team ? team.flag : '🏳️';
}

function renderBracket(data) {
  const container = document.getElementById('bracket-grid-container');
  container.innerHTML = '';

  const r16Col = createBracketColumn('Round of 16');
  data.roundOf16.forEach(m => {
    r16Col.appendChild(createMatchCard(m, 'roundOf16'));
  });
  container.appendChild(r16Col);

  const qfCol = createBracketColumn('Quarterfinals');
  data.quarterfinals.forEach(m => {
    qfCol.appendChild(createMatchCard(m, 'quarterfinals'));
  });
  container.appendChild(qfCol);

  const sfCol = createBracketColumn('Semifinals');
  data.semifinals.forEach(m => {
    sfCol.appendChild(createMatchCard(m, 'semifinals'));
  });
  container.appendChild(sfCol);

  const finalCol = createBracketColumn('Final');
  finalCol.appendChild(createMatchCard(data.final, 'final'));
  container.appendChild(finalCol);

  const champCol = document.createElement('div');
  champCol.className = 'bracket-column champion-column';
  champCol.innerHTML = `<div class="bracket-column-title">Champion</div>`;

  const champCard = document.createElement('div');
  champCard.className = 'champion-card';
  
  if (data.final.winner) {
    champCard.innerHTML = `
      <div class="champion-display">
        <div class="champion-trophy-badge">🏆</div>
        <h3 style="margin-top: 8px;">Winner</h3>
        <span class="champion-flag">${getTeamFlag(data.final.winner)}</span>
        <span class="champion-name">${data.final.winner}</span>
      </div>
    `;
  } else {
    champCard.innerHTML = `
      <div class="champion-display">
        <div class="champion-trophy-badge grayscale">🏆</div>
        <h3 style="margin-top: 8px;">Winner</h3>
        <span class="champion-flag">❓</span>
        <span class="champion-name">TBD</span>
      </div>
    `;
  }
  champCol.appendChild(champCard);
  container.appendChild(champCol);
}

function createBracketColumn(title) {
  const col = document.createElement('div');
  col.className = 'bracket-column';
  col.innerHTML = `<div class="bracket-column-title">${title}</div>`;
  return col;
}

function getMatchDate(id) {
  const dates = {
    'r16-1': 'Jun 28', 'r16-2': 'Jun 28', 'r16-3': 'Jun 29', 'r16-4': 'Jun 29',
    'r16-5': 'Jun 30', 'r16-6': 'Jun 30', 'r16-7': 'Jul 01', 'r16-8': 'Jul 01',
    'qf-1': 'Jul 04', 'qf-2': 'Jul 04', 'qf-3': 'Jul 05', 'qf-4': 'Jul 05',
    'sf-1': 'Jul 09', 'sf-2': 'Jul 10',
    'f-1': 'Jul 19'
  };
  return dates[id] || 'TBD';
}

function createMatchCard(match, stage) {
  const card = document.createElement('div');
  card.className = 'matchup-container';
  
  const team1Class = match.winner === match.team1 ? 'winner' : (match.winner ? 'loser' : '');
  const team2Class = match.winner === match.team2 ? 'winner' : (match.winner ? 'loser' : '');

  const score1 = match.winner ? (match.winner === match.team1 ? '2' : '1') : '-';
  const score2 = match.winner ? (match.winner === match.team2 ? '2' : '1') : '-';

  const team1Content = match.team1 
    ? `<span class="matchup-flag">${getTeamFlag(match.team1)}</span> <span class="matchup-team-name">${match.team1}</span>` 
    : `<span class="matchup-team-name text-muted">TBD</span>`;
  const team2Content = match.team2 
    ? `<span class="matchup-flag">${getTeamFlag(match.team2)}</span> <span class="matchup-team-name">${match.team2}</span>` 
    : `<span class="matchup-team-name text-muted">TBD</span>`;

  const matchLabel = match.id.toUpperCase().replace('-', ' ');
  const matchDate = getMatchDate(match.id);

  card.innerHTML = `
    <div class="matchup-header">
      <span class="matchup-title">${matchLabel}</span>
      <span class="matchup-date">${matchDate}</span>
    </div>
    <div class="matchup-slot ${team1Class}" data-team="1">
      <div class="matchup-slot-team">${team1Content}</div>
      <div class="matchup-score-box">${score1}</div>
    </div>
    <div class="matchup-slot ${team2Class}" data-team="2">
      <div class="matchup-slot-team">${team2Content}</div>
      <div class="matchup-score-box">${score2}</div>
    </div>
  `;

  card.querySelectorAll('.matchup-slot').forEach(slot => {
    slot.addEventListener('click', () => {
      const selectedNum = slot.dataset.team;
      const winner = selectedNum === '1' ? match.team1 : match.team2;
      if (!winner) return;
      setWinner(stage, match.id, winner);
    });
  });

  return card;
}

function setWinner(stage, matchId, winner) {
  if (!currentBracket) return;

  let loser = '';
  if (stage === 'roundOf16') {
    const match = currentBracket.roundOf16.find(m => m.id === matchId);
    match.winner = winner;
    match.loser = winner === match.team1 ? match.team2 : match.team1;
    loser = match.loser;
    const matchIndex = currentBracket.roundOf16.indexOf(match);
    const nextMatchIndex = Math.floor(matchIndex / 2);
    if (matchIndex % 2 === 0) {
      currentBracket.quarterfinals[nextMatchIndex].team1 = winner;
    } else {
      currentBracket.quarterfinals[nextMatchIndex].team2 = winner;
    }
  } else if (stage === 'quarterfinals') {
    const match = currentBracket.quarterfinals.find(m => m.id === matchId);
    match.winner = winner;
    match.loser = winner === match.team1 ? match.team2 : match.team1;
    loser = match.loser;
    const matchIndex = currentBracket.quarterfinals.indexOf(match);
    const nextMatchIndex = Math.floor(matchIndex / 2);
    if (matchIndex % 2 === 0) {
      currentBracket.semifinals[nextMatchIndex].team1 = winner;
    } else {
      currentBracket.semifinals[nextMatchIndex].team2 = winner;
    }
  } else if (stage === 'semifinals') {
    const match = currentBracket.semifinals.find(m => m.id === matchId);
    match.winner = winner;
    match.loser = winner === match.team1 ? match.team2 : match.team1;
    loser = match.loser;
    const matchIndex = currentBracket.semifinals.indexOf(match);
    if (matchIndex === 0) {
      currentBracket.final.team1 = winner;
    } else {
      currentBracket.final.team2 = winner;
    }
  } else if (stage === 'final') {
    currentBracket.final.winner = winner;
    currentBracket.final.loser = winner === currentBracket.final.team1 ? currentBracket.final.team2 : currentBracket.final.team1;
    loser = currentBracket.final.loser;
  }

  fetch('http://localhost:3000/api/bracket', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(currentBracket)
  })
  .then(res => res.json())
  .then(() => {
    renderBracket(currentBracket);
  });
}

document.getElementById('cli-update-btn').addEventListener('click', () => {
  const logDisplay = document.getElementById('cli-log-display');
  logDisplay.textContent = 'Executing CLI Call...';
  
  fetch('http://localhost:3000/api/bracket/update', { method: 'POST' })
    .then(res => res.json())
    .then(result => {
      if (result.status === 'success') {
        logDisplay.textContent = `CLI output: ${result.log}`;
        currentBracket = result.data;
        renderBracket(result.data);
      } else {
        logDisplay.textContent = `CLI error: ${result.message}`;
      }
    })
    .catch(err => {
      logDisplay.textContent = `Request failed: ${err.message}`;
    });
});

document.getElementById('cli-reset-btn').addEventListener('click', () => {
  const logDisplay = document.getElementById('cli-log-display');
  logDisplay.textContent = 'Resetting...';

  const emptyData = {
    roundOf16: [
      { id: 'r16-1', team1: 'Argentina', team2: 'Mexico', winner: 'Argentina', loser: 'Mexico' },
      { id: 'r16-2', team1: 'Brazil', team2: 'USA', winner: 'Brazil', loser: 'USA' },
      { id: 'r16-3', team1: 'France', team2: 'Canada', winner: 'France', loser: 'Canada' },
      { id: 'r16-4', team1: 'England', team2: 'Morocco', winner: 'England', loser: 'Morocco' },
      { id: 'r16-5', team1: 'Spain', team2: 'Japan', winner: 'Spain', loser: 'Japan' },
      { id: 'r16-6', team1: 'Portugal', team2: 'South Korea', winner: 'Portugal', loser: 'South Korea' },
      { id: 'r16-7', team1: 'Germany', team2: 'Australia', winner: 'Germany', loser: 'Australia' },
      { id: 'r16-8', team1: 'Italy', team2: 'Saudi Arabia', winner: 'Italy', loser: 'Saudi Arabia' }
    ],
    quarterfinals: [
      { id: 'qf-1', team1: 'Argentina', team2: 'Brazil', winner: null, loser: null },
      { id: 'qf-2', team1: 'France', team2: 'England', winner: null, loser: null },
      { id: 'qf-3', team1: 'Spain', team2: 'Portugal', winner: null, loser: null },
      { id: 'qf-4', team1: 'Germany', team2: 'Italy', winner: null, loser: null }
    ],
    semifinals: [
      { id: 'sf-1', team1: '', team2: '', winner: null, loser: null },
      { id: 'sf-2', team1: '', team2: '', winner: null, loser: null }
    ],
    final: { id: 'f-1', team1: '', team2: '', winner: null, loser: null }
  };

  fetch('http://localhost:3000/api/bracket', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(emptyData)
  })
  .then(res => res.json())
  .then(() => {
    currentBracket = emptyData;
    renderBracket(emptyData);
    logDisplay.textContent = 'Bracket has been reset.';
  });
});

const chancesContainer = document.getElementById('chances-list-container');
const predictorTeam1 = document.getElementById('predictor-team1');
const predictorTeam2 = document.getElementById('predictor-team2');
const predictBtn = document.getElementById('predict-btn');
const predictionResults = document.getElementById('prediction-results-container');

function renderWinningChances() {
  chancesContainer.innerHTML = '';
  const sorted = [...teamsData].sort((a, b) => b.chance - a.chance);
  
  sorted.forEach(team => {
    const item = document.createElement('div');
    item.className = 'chance-item';
    item.innerHTML = `
      <div class="chance-info">
        <span class="chance-team">${team.flag} ${team.name}</span>
        <span class="chance-val">${team.chance}%</span>
      </div>
      <div class="chance-bar-container">
        <div class="chance-bar" style="width: ${team.chance * 5}%"></div>
      </div>
    `;
    chancesContainer.appendChild(item);
  });

  populatePredictorSelects();
}

function populatePredictorSelects() {
  if (predictorTeam1.children.length > 0) return;

  const sorted = [...teamsData].sort((a, b) => a.name.localeCompare(b.name));
  sorted.forEach(team => {
    const opt1 = document.createElement('option');
    opt1.value = team.id;
    opt1.textContent = `${team.flag} ${team.name}`;
    predictorTeam1.appendChild(opt1);

    const opt2 = document.createElement('option');
    opt2.value = team.id;
    opt2.textContent = `${team.flag} ${team.name}`;
    predictorTeam2.appendChild(opt2);
  });

  if (predictorTeam2.children.length > 1) {
    predictorTeam2.selectedIndex = 1;
  }
}

predictBtn.addEventListener('click', () => {
  const team1Id = predictorTeam1.value;
  const team2Id = predictorTeam2.value;

  if (team1Id === team2Id) {
    predictionResults.innerHTML = `
      <div class="results-grid" style="grid-template-columns: 1fr; text-align: center; color: var(--color-danger); font-weight: 700;">
        Please select two different teams.
      </div>
    `;
    return;
  }

  const team1 = teamsData.find(t => t.id === team1Id);
  const team2 = teamsData.find(t => t.id === team2Id);

  const baseChance1 = team1.chance + (team1.titles * 2);
  const baseChance2 = team2.chance + (team2.titles * 2);
  const total = baseChance1 + baseChance2;

  const pct1 = Math.round((baseChance1 / total) * 100);
  const pct2 = 100 - pct1;

  const winner = pct1 > pct2 ? team1 : (pct2 > pct1 ? team2 : null);

  predictionResults.innerHTML = `
    <div class="predictor-results-card">
      <div class="predictor-result-header">
        <span class="result-team-info">${team1.flag} ${team1.name} (${pct1}%)</span>
        <span class="vs-label">VS</span>
        <span class="result-team-info">${team2.name} ${team2.flag} (${pct2}%)</span>
      </div>
      <div class="comparison-odds-bar">
        <div class="odds-team-a" style="width: ${pct1}%"></div>
        <div class="odds-team-b" style="width: ${pct2}%"></div>
      </div>
      <div class="predictor-result-percentages">
        <span>${pct1}%</span>
        <span>${pct2}%</span>
      </div>
    </div>
  `;
});

window.addEventListener('DOMContentLoaded', () => {
  renderTeams();
  if (teamsData.length > 0) {
    showTeamDetails(getSortedTeams()[0]);
    const firstItem = teamsListContainer.querySelector('.team-item');
    if (firstItem) firstItem.classList.add('active');
  }

  fetch('http://localhost:3000/api/bracket/update', { method: 'POST' })
    .then(res => res.json())
    .then(result => {
      if (result.status === 'success') {
        currentBracket = result.data;
      }
    })
    .catch(() => {});
  
  renderNonQualifiers();
});

const nonQualifiersData = [
  { id: 'italy', name: 'Italy', flag: '🇮🇹', titles: 4, rank: 12, reason: 'Failed in UEFA playoffs', star: 'Gianluigi Donnarumma' },
  { id: 'nigeria', name: 'Nigeria', flag: '🇳🇬', titles: 0, rank: 26, reason: 'Failed in CAF qualification', star: 'Victor Osimhen' },
  { id: 'denmark', name: 'Denmark', flag: '🇩🇰', titles: 0, rank: 20, reason: 'Missed UEFA qualification', star: 'Christian Eriksen' },
  { id: 'poland', name: 'Poland', flag: '🇵🇱', titles: 0, rank: 35, reason: 'Eliminated in qualifiers', star: 'Robert Lewandowski' },
  { id: 'russia', name: 'Russia', flag: '🇷🇺', titles: 0, rank: 38, reason: 'Suspended by FIFA/UEFA', star: 'Aleksandr Golovin' }
];

function renderNonQualifiers() {
  const container = document.getElementById('non-qualifiers-container');
  if (!container) return;
  
  container.innerHTML = nonQualifiersData.map(team => `
    <div class="nq-card">
      <div class="nq-header">
        <span class="nq-flag">${team.flag}</span>
        <h3 class="nq-name">${team.name}</h3>
      </div>
      <div class="nq-stats">
        <div class="nq-stat"><span>World Rank:</span> <strong>#${team.rank}</strong></div>
        <div class="nq-stat"><span>World Cups:</span> <strong>${team.titles} 🏆</strong></div>
        <div class="nq-stat"><span>Top Star:</span> <strong>${team.star}</strong></div>
        <div class="nq-reason">${team.reason}</div>
      </div>
    </div>
  `).join('');
}
