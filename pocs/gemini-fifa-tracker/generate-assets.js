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

const dishDescriptions = {
  'Poutine': 'Crispy fries with cheese curds & rich brown gravy.',
  'Butter Tarts': 'Pastry shells filled with sweet butter & sugar.',
  'Tourtière': 'French-Canadian meat pie with spiced pork & beef.',
  'Tacos': 'Corn tortillas folded with seasoned meats & cilantro.',
  'Mole Poblano': 'Chicken in a rich chocolate & chili pepper sauce.',
  'Chiles en Nogada': 'Stuffed chilies topped with walnut sauce & pome.',
  'Hamburger': 'Beef patty in a bun with fresh cheese & toppings.',
  'Apple Pie': 'Sweet double-crust pie with spiced apple filling.',
  'Clam Chowder': 'Creamy New England soup with clams & potatoes.',
  'Wiener Schnitzel': 'Pan-fried breaded veal cutlet served with lemon.',
  'Sachertorte': 'Decadent Austrian double-layer chocolate cake.',
  'Apfelstrudel': 'Warm pastry filled with spiced sliced apples.',
  'Moules-Frites': 'Steamed mussels served with gold crispy fries.',
  'Belgian Waffles': 'Fluffy sweet waffles topped with sugar crystals.',
  'Carbonnade Flamande': 'Sweet-sour beef & onion stew cooked in beer.',
  'Ćevapi': 'Grilled minced meat sausages with flatbread.',
  'Burek': 'Flaky pastry sheets filled with meat or cheese.',
  'Klepe': 'Traditional minced meat dumplings boiled & sauced.',
  'Peka': 'Slow-cooked meat & vegetables baked under a dome.',
  'Crni Rižot': 'Rich black risotto cooked with cuttlefish ink.',
  'Fritule': 'Mini sweet donuts flavoured with brandy & raisins.',
  'Vepřo Knedlo Zelo': 'Roasted pork loin served with dumplings & cabbage.',
  'Svíčková': 'Beef tenderloin in vegetable cream sauce with cran.',
  'Trdelník': 'Spit-roasted sweet dough rolled in cinnamon sugar.',
  'Fish and Chips': 'Crispy battered fish fillets served with hot fries.',
  'Sunday Roast': 'Roasted meat served with Yorkshire pudding & gravy.',
  'Shepherd\'s Pie': 'Minced lamb cooked in gravy topped with mashed pot.',
  'Coq au Vin': 'Chicken braised in red wine, bacon & mushrooms.',
  'Ratatouille': 'Stewed summer vegetables with zucchini & eggplant.',
  'Crème Brûlée': 'Rich custard dessert topped with caramelized sugar.',
  'Bratwurst': 'German grilled sausage served with mustard & kraut.',
  'Sauerkraut': 'Fermented finely cut cabbage with a sour flavor.',
  'Pretzel': 'Baked knot-shaped pastry sprinkled with coarse salt.',
  'Stroopwafel': 'Thin waffle cookies joined by caramel syrup filling.',
  'Bitterballen': 'Deep-fried ragout balls served with mustard.',
  'Stamppot': 'Mashed potatoes mixed with kale & smoked sausage.',
  'Fårikål': 'Traditional mutton & cabbage stew with black pepper.',
  'Gravlaks': 'Dry-cured salmon seasoned with dill & salt.',
  'Lutefisk': 'Traditional dried whitefish treated with lye.',
  'Bacalhau à Brás': 'Shredded salt cod cooked with onions & fries.',
  'Pastel de Nata': 'Portuguese custard tart dusted with cinnamon.',
  'Caldo Verde': 'Green soup made of potatoes, collards & chorizo.',
  'Haggis': 'Savoury pudding with sheep\'s pluck, oats & spices.',
  'Scotch Pie': 'Double-crust meat pie filled with minced mutton.',
  'Cranachan': 'Raspberry, toasted oats, cream & whisky dessert.',
  'Paella': 'Saffron rice cooked with seafood & fresh vegetables.',
  'Tortilla Española': 'Thick Spanish omelette cooked with sliced potatoes.',
  'Gazpacho': 'Chilled tomato soup blended with cucumber & olive oil.',
  'Köttbullar': 'Swedish meatballs served with gravy & lingonberries.',
  'Gravlax': 'Cured raw salmon infused with fresh dill & sugars.',
  'Smörgåstårta': 'Savoury sandwich cake layered with seafood fillings.',
  'Fondue': 'Melted cheese blend served hot in a communal pot.',
  'Raclette': 'Melted cheese scraped over potatoes & cured meats.',
  'Rösti': 'Crispy pan-fried shredded potato pancake.',
  'Kebab': 'Skewered grilled meat seasoned with Middle East spices.',
  'Baklava': 'Layered phyllo pastry filled with chopped nuts & honey.',
  'Pide': 'Flatbread baked with minced meat, spinach & cheese.',
  'Asado': 'Traditional grilled beef ribs cooked over wood fire.',
  'Empanadas': 'Baked pastry turnovers stuffed with savory beef filling.',
  'Dulce de Leche': 'Sweet caramelized milk spread used in desserts.',
  'Feijoada': 'Rich black bean stew simmered with pork & beef cuts.',
  'Pão de Queijo': 'Chewy cheese rolls made of tapioca flour.',
  'Brigadeiro': 'Sweet chocolate truffles coated in sprinkles.',
  'Bandeja Paisa': 'Platter of beans, rice, pork belly, egg & plantain.',
  'Arepas': 'Round cornmeal patties grilled and stuffed with fillings.',
  'Ajiaco': 'Thick chicken potato soup cooked with guascas herb.',
  'Ceviche': 'Fresh raw fish cured in citrus juices with red onions.',
  'Llapingachos': 'Pan-fried potato cakes stuffed with cheese.',
  'Locro de Papa': 'Creamy potato & cheese soup flavoured with annatto.',
  'Sopa Paraguaya': 'Savory cornbread baked with cheese & onion.',
  'Chipa': 'Baked cheese bread rings made of cassava flour.',
  'Mbejú': 'Starchy pan-baked flatbread made of starch & cheese.',
  'Chivito': 'Steak sandwich layered with ham, cheese, egg & bacon.',
  'Martín Fierro': 'Sweet dessert slice of quince paste & cheese.',
  'Meat Pie': 'Shortcrust pastry pie filled with minced meat gravy.',
  'Vegemite Toast': 'Toasted bread spread with savory yeast paste.',
  'Pavlova': 'Crisp meringue dessert topped with whipped cream.',
  'Chelo Kebab': 'Skewered minced meat served over saffron basmati rice.',
  'Ghormeh Sabzi': 'Herb stew cooked with kidney beans, lamb & lime.',
  'Fesenjan': 'Sweet-tart stew made of pomegranate & walnuts.',
  'Masgouf': 'Seasoned carp fish slow-grilled over open wood fire.',
  'Biryani': 'Fragrant spiced long-grain rice layered with chicken.',
  'Kleicha': 'Traditional Iraqi date cookies scented with cardamom.',
  'Sushi': 'Vinegared rice rolled with seaweed & fresh raw fish.',
  'Ramen': 'Noodle soup served in savory broth with sliced pork.',
  'Tempura': 'Deep-fried seafood & vegetables in light crispy batter.',
  'Mansaf': 'Lamb cooked in dried yogurt sauce, served with rice.',
  'Falafel': 'Deep-fried spiced ground chickpea herb balls.',
  'Kanafeh': 'Sweet spun pastry layered with cheese and rose syrup.',
  'Machboos': 'Spiced mixed rice dish cooked with chicken or mutton.',
  'Luqaimat': 'Sweet crunchy dumpling balls glazed in syrup.',
  'Harees': 'Slow-cooked wheat & meat porridge topped with butter.',
  'Kabsa': 'Mixed rice dish cooked with meat, vegetables & cardamom.',
  'Jareesh': 'Crushed wheat cooked with yogurt & chicken.',
  'Mutabbaq': 'Folded pan-fried thin pancake stuffed with beef.',
  'Kimchi': 'Spiced fermented cabbage seasoned with chili & garlic.',
  'Bulgogi': 'Thinly sliced beef marinated in soy & pear juice.',
  'Bibimbap': 'Warm rice bowl topped with seasoned vegetables & egg.',
  'Plov': 'Uzbek rice pilaf simmered with lamb, carrots & cumin.',
  'Somsa': 'Baked flaky pastry pocket stuffed with minced lamb.',
  'Lagman': 'Hand-pulled noodles served in lamb vegetable broth.',
  'Couscous': 'Steamed semolina granules served with vegetable stew.',
  'Shakshouka': 'Poached eggs cooked in a spicy tomato pepper sauce.',
  'Tajine': 'Slow-cooked meat & fruit stew prepared in clay pot.',
  'Cachupa': 'Slow-cooked stew of corn, beans, cassava & sausage.',
  'Pastel': 'Deep-fried savory pastry pocket filled with fish.',
  'Pudim de Leite': 'Creamy baked condensed milk caramel flan.',
  'Garba': 'Fried plantains served with cassava semolina & tuna.',
  'Aloko': 'Fried ripe plantains seasoned with pepper onion sauce.',
  'Kedjenou': 'Chicken stew slow-cooked in sealed clay pot.',
  'Moambé Chicken': 'Chicken cooked in rich palm butter sauce & spices.',
  'Fufu': 'Thick cassava dough used to scoop stews.',
  'Chikwangue': 'Cassava dough wrapped in banana leaves & steamed.',
  'Koshary': 'Mixed rice, macaroni, lentils, topped with tomato sauce.',
  'Ful Medames': 'Warm fava beans cooked with garlic, lemon & oil.',
  'Mulukhiyah': 'Jute leaves soup cooked with garlic, coriander & chicken.',
  'Jollof Rice': 'One-pot rice dish simmered in spicy tomato stew.',
  'Kelewele': 'Spicy fried plantain cubes seasoned with ginger.',
  'Tagine': 'Slow-cooked meat stew with prunes and almonds.',
  'Harira': 'Hearty tomato, lentil, chickpea, & herb soup.',
  'Thiéboudienne': 'Senegalese rice simmered in tomato paste with fish.',
  'Yassa Poulet': 'Chicken marinated in caramelized onions, lemon & mustard.',
  'Maafe': 'Rich peanut butter sauce stew cooked with beef & carrots.',
  'Biltong': 'Air-dried cured spiced meat strips.',
  'Bobotie': 'Spiced minced meat bake topped with savory egg custard.',
  'Bunny Chow': 'Hollowed loaf of bread filled with spicy mutton curry.',
  'Brik': 'Crispy deep-fried pastry sheet enclosing a whole egg.',
  'Lablabi': 'Garlicky chickpea soup served over stale crusty bread.',
  'Keshi Yena': 'Steamed cheese shell stuffed with chicken & raisins.',
  'Stobá': 'Hearty beef stew cooked with potatoes and vegetables.',
  'Sopito': 'Creamy coconut milk fish soup with fresh lime.',
  'Griot': 'Crispy marinated fried pork chunks served with pikliz.',
  'Soup Joumou': 'Traditional squash soup celebrating Haiti independence.',
  'Akasan': 'Sweet warm cornmeal shake flavored with cinnamon & milk.',
  'Sancocho': 'Thick root vegetable & chicken soup flavored with cilantro.',
  'Ropa Vieja': 'Shredded beef braised in sweet tomato pepper sauce.',
  'Carimañolas': 'Fried cassava fritters stuffed with seasoned ground beef.',
  'Hāngī': 'Traditional Maori underground steam-cooked feast platter.',
  'Pavlova': 'Crisp baked meringue dessert topped with fresh berries.',
  'Whitebait Fritter': 'Delicate pan-fried egg patties with tiny whitebait fish.'
};

const goalkeepers = ['Petr Čech', 'Craig Forrest', 'Jose Luis Chilavert', 'Essam El-Hadary', 'Rüştü Reçber', 'Iker Casillas', 'Erik Thorstvedt', 'Robert Kidiaba', 'Johnny Placide'];
const defenders = ['Franz Beckenbauer', 'Virgil van Dijk', 'Andrew Robertson', 'Lucas Radebe', 'Ryan Nelsen', 'Cuco Martina', 'Vincent Kompany', 'Rafael Márquez', 'Bobby Moore', 'Noureddine Naybet', 'Radhi Jaïdi', 'Ivan Vicelich'];
const forwards = [
  'Hugo Sánchez', 'Javier Hernández', 'Landon Donovan', 'Gary Lineker', 'Zinedine Zidane', 'Michel Platini',
  'Thierry Henry', 'Gerd Müller', 'Miroslav Klose', 'Johan Cruyff', 'Marco van Basten', 'Cristiano Ronaldo',
  'Eusébio', 'Kenny Dalglish', 'Diego Maradona', 'Lionel Messi', 'Pelé', 'Ronaldo', 'Ronaldinho', 'Radamel Falcao',
  'Luis Suárez', 'Diego Forlán', 'Tim Cahill', 'Ali Daei', 'Kaoru Mitoma', 'Chris Wood', 'Riyad Mahrez',
  'Didier Drogba', 'Sébastien Haller', 'Mohamed Salah', 'Sadio Mané', 'Benni McCarthy', 'Julio Dely Valdés',
  'Blas Pérez', 'Luis Tejada', 'Wynton Rufer', 'Alphonso Davies', 'Santiago Giménez', 'Christian Pulisic',
  'Konrad Laimer', 'Patrik Schick', 'Erling Haaland', 'Alexander Isak', 'Luis Díaz', 'Moisés Caicedo',
  'Julio Enciso', 'Federico Valverde', 'Nestory Irankunda', 'Mehdi Taremi', 'Aymen Hussein', 'Mousa Al-Tamari',
  'Akram Afif', 'Salem Al-Dawsari', 'Son Heung-min', 'Eldor Shomurodov', 'Ryan Mendes', 'Chancel Mbemba',
  'Mohammed Kudus', 'Percy Tau', 'Ellyes Skhiri', 'Juninho Bacuna', 'Frantzdy Pierrot', 'Adalberto Carrasquilla'
];

function getPlayerStats(name, isStar) {
  const cleanName = name.trim();
  let pos = 'MID';
  if (goalkeepers.includes(cleanName)) pos = 'GK';
  else if (defenders.includes(cleanName)) pos = 'DEF';
  else if (forwards.includes(cleanName)) pos = 'FWD';

  const baseRating = isStar ? 91 : 88;
  const ratingOffset = (cleanName.length % 5) - 2;
  const rating = baseRating + ratingOffset;
  const number = (cleanName.length * 3) % 22 + 1;

  if (pos === 'GK') {
    return {
      pos, rating, number,
      labels: ['DIV', 'HAN', 'KIC', 'REF', 'SPD', 'POS'],
      values: [rating + 2, rating - 1, rating - 3, rating + 4, rating - 5, rating + 1]
    };
  } else if (pos === 'DEF') {
    return {
      pos, rating, number,
      labels: ['PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY'],
      values: [rating - 8, rating - 35, rating - 10, rating - 12, rating + 4, rating + 3]
    };
  } else if (pos === 'FWD') {
    return {
      pos, rating, number,
      labels: ['PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY'],
      values: [rating + 3, rating + 4, rating - 12, rating + 1, rating - 45, rating - 10]
    };
  } else {
    return {
      pos, rating, number,
      labels: ['PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY'],
      values: [rating - 2, rating - 10, rating + 4, rating + 3, rating - 25, rating - 15]
    };
  }
}

const publicDir = path.join(process.cwd(), 'public');
const assetsDir = path.join(publicDir, 'assets');
const playersDir = path.join(assetsDir, 'players');
const dishesDir = path.join(assetsDir, 'dishes');

if (!fs.existsSync(assetsDir)) fs.mkdirSync(assetsDir, { recursive: true });
if (!fs.existsSync(playersDir)) fs.mkdirSync(playersDir, { recursive: true });
if (!fs.existsSync(dishesDir)) fs.mkdirSync(dishesDir, { recursive: true });

function getPlayerCardSVG(name, type, flag, color1, color2) {
  const isStar = type === 'star';
  const stats = getPlayerStats(name, isStar);
  const cardGradStart = isStar ? '#f59e0b' : '#cbd5e1';
  const cardGradEnd = isStar ? '#0f172a' : '#1e293b';
  const borderColor = isStar ? '#fbbf24' : '#94a3b8';
  const safeName = name.replace(/'/g, "\\'");
  
  const textColor = (color1 === '#ffffff' || color1 === '#fff') ? '#0f172a' : '#ffffff';
  
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
  <text x="157.5" y="30" font-family="'Outfit', sans-serif" font-size="8" font-weight="800" fill="#0f172a" text-anchor="middle">${stats.pos}</text>
  
  <text x="25" y="52" font-family="'Outfit', sans-serif" font-size="16" font-weight="800" fill="#ffffff" text-anchor="middle">${stats.rating}</text>
  
  <g transform="translate(0, 20)">
    <circle cx="100" cy="90" r="14" fill="#ffffff" fill-opacity="0.9" />
    <path d="M 85,110 L 70,110 L 45,135 L 58,145 L 75,130 L 75,175 L 125,175 L 125,130 L 142,145 L 155,135 L 130,110 L 115,110 Z" fill="url(#jerseyGrad-${name.replace(/[^a-zA-Z0-9]/g, '')})" />
    <text x="100" y="162" font-family="'Outfit', sans-serif" font-size="7" font-weight="800" fill="${textColor}" text-anchor="middle">${name.split(' ').pop().toUpperCase()}</text>
    <text x="100" y="150" font-family="'Outfit', sans-serif" font-size="18" font-weight="900" fill="${textColor}" text-anchor="middle">${stats.number}</text>
  </g>
  
  <rect x="15" y="220" width="170" height="45" rx="8" fill="#000000" fill-opacity="0.5" stroke="${borderColor}" stroke-opacity="0.3" />
  <text x="100" y="238" font-family="'Outfit', sans-serif" font-size="11" font-weight="800" fill="#ffffff" text-anchor="middle">${safeName}</text>
  
  <g transform="translate(20, 245)" font-family="'Outfit', sans-serif" font-size="7" font-weight="700" fill="#cbd5e1" text-anchor="middle">
    <text x="10" y="5">${stats.labels[0]}</text><text x="10" y="13" fill="#ffffff">${stats.values[0]}</text>
    <text x="38" y="5">${stats.labels[1]}</text><text x="38" y="13" fill="#ffffff">${stats.values[1]}</text>
    <text x="66" y="5">${stats.labels[2]}</text><text x="66" y="13" fill="#ffffff">${stats.values[2]}</text>
    <text x="94" y="5">${stats.labels[3]}</text><text x="94" y="13" fill="#ffffff">${stats.values[3]}</text>
    <text x="122" y="5">${stats.labels[4]}</text><text x="122" y="13" fill="#ffffff">${stats.values[4]}</text>
    <text x="150" y="5">${stats.labels[5]}</text><text x="150" y="13" fill="#ffffff">${stats.values[5]}</text>
  </g>
</svg>`;
}

function getDishCardSVG(name, flag) {
  const safeName = name.replace(/'/g, "\\'");
  const desc = dishDescriptions[name] || 'A traditional local dish from this region.';
  
  let dishGraphics = '';
  if (name.includes('Tacos') || name.includes('Empanadas') || name.includes('Somsa') || name.includes('Burek') || name.includes('Mutabbaq')) {
    dishGraphics = `
      <ellipse cx="0" cy="20" rx="60" ry="15" fill="#e2e8f0" stroke="#cbd5e1" stroke-width="2" />
      <path d="M -45,10 Q 0,-25 45,10 Z" fill="#eab308" stroke="#ca8a04" stroke-width="2" />
      <circle cx="-15" cy="5" r="5" fill="#ef4444" />
      <circle cx="15" cy="8" r="4" fill="#22c55e" />
      <circle cx="0" cy="0" r="6" fill="#ca8a04" />
    `;
  } else if (name.includes('Hamburger') || name.includes('Burger') || name.includes('Chivito')) {
    dishGraphics = `
      <ellipse cx="0" cy="22" rx="55" ry="12" fill="#e2e8f0" stroke="#cbd5e1" stroke-width="2" />
      <path d="M -35,5 Q 0,-30 35,5 Z" fill="#f59e0b" />
      <rect x="-38" y="5" width="76" height="8" rx="2" fill="#7c2d12" />
      <path d="M -40,13 L 40,13 L 30,19 L -30,19 Z" fill="#eab308" />
      <path d="M -35,19 Q 0,30 35,19 Z" fill="#f59e0b" />
    `;
  } else if (name.includes('Poutine') || name.includes('Frites') || name.includes('Chips')) {
    dishGraphics = `
      <ellipse cx="0" cy="20" rx="60" ry="15" fill="#e2e8f0" stroke="#cbd5e1" stroke-width="2" />
      <rect x="-35" y="8" width="10" height="20" rx="2" fill="#fbbf24" transform="rotate(15)" />
      <rect x="-15" y="5" width="8" height="24" rx="2" fill="#fbbf24" transform="rotate(-5)" />
      <rect x="5" y="8" width="9" height="22" rx="2" fill="#fbbf24" transform="rotate(30)" />
      <rect x="25" y="6" width="8" height="20" rx="2" fill="#fbbf24" transform="rotate(-15)" />
      <path d="M -45,15 Q 0,-5 45,15" fill="none" stroke="#78350f" stroke-width="6" stroke-linecap="round" opacity="0.8" />
      <circle cx="-10" cy="12" r="5" fill="#f8fafc" stroke="#cbd5e1" />
      <circle cx="15" cy="14" r="5" fill="#f8fafc" stroke="#cbd5e1" />
    `;
  } else if (name.includes('Soup') || name.includes('Chowder') || name.includes('Stew') || name.includes('Koshary') || name.includes('Ramen') || name.includes('Tajine') || name.includes('Cachupa') || name.includes('Harira') || name.includes('Ghormeh') || name.includes('Ajiaco') || name.includes('Lablabi')) {
    dishGraphics = `
      <ellipse cx="0" cy="25" rx="55" ry="10" fill="#e2e8f0" stroke="#cbd5e1" />
      <path d="M -40,15 C -40,45 40,45 40,15 Z" fill="#be123c" stroke="#9f1239" stroke-width="2" />
      <ellipse cx="0" cy="15" rx="40" ry="10" fill="#f59e0b" />
      <circle cx="-10" cy="15" r="4" fill="#ef4444" />
      <circle cx="15" cy="17" r="5" fill="#22c55e" />
      <path d="M -15,5 Q -10,-10 -15,-20" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" opacity="0.5" />
      <path d="M 0,5 Q 5,-15 0,-25" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" opacity="0.7" />
      <path d="M 15,5 Q 20,-10 15,-20" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" opacity="0.5" />
    `;
  } else if (name.includes('Pie') || name.includes('Sachertorte') || name.includes('Strudel') || name.includes('Waffles') || name.includes('Baklava') || name.includes('Pavlova') || name.includes('Tarts') || name.includes('Brigadeiro') || name.includes('Nata')) {
    dishGraphics = `
      <ellipse cx="0" cy="22" rx="55" ry="12" fill="#e2e8f0" stroke="#cbd5e1" stroke-width="2" />
      <ellipse cx="0" cy="15" rx="45" ry="12" fill="#db2777" />
      <path d="M -45,15 Q 0,-20 45,15 Z" fill="#f472b6" opacity="0.9" />
      <circle cx="0" cy="-5" r="6" fill="#dc2626" />
      <path d="M 0,-5 Q 10,-15 20,-10" fill="none" stroke="#16a34a" stroke-width="2" stroke-linecap="round" />
    `;
  } else {
    dishGraphics = `
      <ellipse cx="0" cy="24" rx="65" ry="12" fill="#334155" opacity="0.5" />
      <ellipse cx="0" cy="18" rx="60" ry="15" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="2" />
      <path d="M -45,10 C -30,35 30,35 45,10 Z" fill="#d97706" opacity="0.9" />
      <ellipse cx="0" cy="8" rx="35" ry="8" fill="#eab308" />
      <circle cx="-10" cy="8" r="4" fill="#22c55e" />
      <circle cx="10" cy="10" r="4" fill="#ef4444" />
      <path d="M -10,-5 Q -5,-15 -10,-25" fill="none" stroke="#ffffff" stroke-width="2" opacity="0.5" />
      <path d="M 10,-5 Q 15,-15 10,-25" fill="none" stroke="#ffffff" stroke-width="2" opacity="0.5" />
    `;
  }

  return `<svg xmlns="http://www.w3.org/2000/svg" width="300" height="200" viewBox="0 0 300 200">
  <defs>
    <linearGradient id="dishGrad-${name.replace(/[^a-zA-Z0-9]/g, '')}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#334155" />
    </linearGradient>
  </defs>
  
  <rect x="5" y="5" width="290" height="190" rx="16" fill="url(#dishGrad-${name.replace(/[^a-zA-Z0-9]/g, '')})" stroke="#475569" stroke-width="2" />
  
  <g transform="translate(150, 80)">
    ${dishGraphics}
  </g>
  
  <rect x="15" y="140" width="270" height="48" rx="8" fill="#000000" fill-opacity="0.65" stroke="#475569" stroke-opacity="0.5" />
  <text x="150" y="157" font-family="'Outfit', sans-serif" font-size="12" font-weight="800" fill="#ffffff" text-anchor="middle">${safeName}</text>
  <text x="150" y="172" font-family="'Outfit', sans-serif" font-size="8" font-weight="500" fill="#cbd5e1" text-anchor="middle">${desc}</text>
  <text x="35" y="169" font-family="'Outfit', sans-serif" font-size="18" text-anchor="middle">${flag}</text>
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
