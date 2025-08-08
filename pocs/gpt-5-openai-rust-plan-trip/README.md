# ai-agent-planner-rs

A tiny Rust CLI that asks for a destination and number of days and generates a day-by-day travel itinerary using the OpenAI API.

## Prerequisites

- Linux shell
- Rust toolchain (cargo)
- An OpenAI API key with access to a chat model (e.g., gpt-4o-mini)

Export your key in the environment:

```bash
export OPENAI_API_KEY="sk-..."
```

## Build

```bash
cargo build --release
```

## Run

```bash
cargo run
```

### Result

```bash
AI Trip Planner (ai-agent-planner-rs)
Enter destination city (e.g., New York): Las Vegas
How many days will you stay there?: 2

Your plan for Las Vegas (2 days):

2-Day Las Vegas Itinerary (landmarks + hidden gems, easy moving around, great bites)

General tips
- Distances on the Strip are longer than they look; use pedestrian bridges and cut through casinos to stay cool/dry.
- RTC’s Deuce bus runs 24/7 up and down the Strip to Downtown; buy passes in the rideRTC app. The Las Vegas Monorail (MGM Grand ↔ SAHARA) is fast for hops on the east side of the Strip.
- Make key reservations: Raku (Chinatown), Esther’s Kitchen (Arts District), shows (O, KA, Sphere).
- Rain is rare, but many indoor swaps are listed under each time block.

Day 1 – The Strip icons with clever shortcuts
Morning (8:30–12:30)
- 8:30–9:15 Coffee + pastry at Urth Caffé (Wynn Plaza). Good latte art; grab-and-go if eager to explore.
  Tip: Stroll Wynn’s atrium and floral carousel (10–15 min, free, indoors).
- 9:30–11:00 Venetian/Palazzo and Grand Canal Shoppes. Window-shop and watch gondolas (indoor). Optional gondola ride (30–40 min).
  Snack nearby: Black Tap (crazy shakes) or Yardbird for an early, hearty Southern brunch.
- 11:05–12:00 LINQ Promenade walk (10–12 min). Pop into the Flamingo Wildlife Habitat (free, 20–30 min; a quiet, hidden oasis).
  Coffee backup: Gabi Coffee & Bakery is a short rideshare away, but save for Day 2’s Chinatown.

Transit: Walk via bridges. If feet need a break, hop the Monorail from Harrah’s/The LINQ north or south as needed.

Afternoon (12:30–5:30)
- 12:30–1:30 High Roller at The LINQ. Best views by day or at sunset; standard loop ~30 min. Buy timed tickets online to skip lines.
  Quick bites at LINQ: In-N-Out (cheap and fast), Gordon Ramsay Fish & Chips.
- 1:45–3:00 Caesars Forum Shops (indoor). Catch the free Fall of Atlantis animatronic show on the hour (10 min). Window-shop in AC.
  Dessert: Dominique Ansel Las Vegas (Cronut rotation, excellent eclairs) inside Caesars.
- 3:10–4:00 Bellagio Conservatory & Botanical Gardens (free, 30–45 min, indoors).
- 4:00–4:20 Bellagio Patisserie stop. The world’s largest chocolate fountain is here—grab a macaron or gelato.

Transit: Venetian → LINQ (10–12 min walk). LINQ → Caesars → Bellagio is a pleasant indoor-connected route through Forum Shops (keeps you out of sun/rain).

Evening (5:30–late)
- 5:30–7:00 Early dinner near ARIA/Cosmopolitan
  Options:
  - Din Tai Fung (ARIA): xiao long bao and noodles; book ahead (60–90 min).
  - Jaleo (Cosmopolitan): tapas for sharing.
  - Best Friend (Park MGM): Korean-Mex by Roy Choi, casual-fun.
- 7:30–8:00 Bellagio Fountains. Nightly shows every 15–30 min; stand center-lake for symmetry or by the Louis Vuitton store for great angles.
- 8:30–10:00 Choose your show or spectacle
  - Cirque du Soleil: O (Bellagio, aquatic) or KA (MGM Grand, epic staging).
  - The Sphere: “Postcard from Earth” (futuristic, immersive, all indoor).
  - If skipping shows: The Chandelier Bar (Cosmopolitan) for a signature Verbena cocktail and tingling flower.
- 10:00–11:00 Nightcap/dessert
  - Secret Pizza (Cosmo, unmarked corridor on level 3) for a late slice.
  - Eataly (Park MGM) for gelato.
  - Speakeasy vibe: Ghost Donkey (Cosmo) for mezcal and truffle nachos.

Rainy-day swaps for Day 1
- Spend more time indoors at Forum Shops, Grand Canal Shoppes, and Bellagio Conservatory.
- Shark Reef Aquarium (Mandalay Bay) or Area15 (Omega Mart, Wink World) are excellent all-indoor options.
- Skip outdoor Flamingo Habitat if it’s pouring; add Pinball Hall of Fame (near the Welcome Sign, all indoor).

Walking/public transit notes
- Monorail stations: Harrah’s/The LINQ and Bally’s/Paris help you hop quickly without street crowds.
- Free trams: Park MGM ↔ Aria ↔ Bellagio; Mandalay Bay ↔ Luxor ↔ Excalibur.

Day 2 – Downtown, Arts District, neon glow, and Chinatown eats
Morning (9:00–12:30)
- 9:00–10:00 Breakfast + coffee at PublicUs (Downtown). Local favorite; try the honey-lavender latte or a breakfast tartine.
- 10:10–12:00 The Mob Museum (90–120 min). Powerful exhibits; the Crime Lab and Distillery add-on experiences are fun if you have time.
  Alternative/extra: Short stroll through Fremont East for murals and vintage neon (20–30 min).
  Sweet stop nearby: Donut Bar (if it’s open; sells out early) or Evel Pie for a cheeky slice if skipping breakfast.

Transit from Strip: The Deuce bus to Fremont Street (35–50 min depending on traffic). Rideshare is 15–20 min.

Afternoon (12:30–5:30)
- 12:30–2:00 Lunch in the Arts District (18b)
  - Esther’s Kitchen (house-made pasta, bread program is stellar; reserve).
  - Good Pie (Detroit-style square slices).
  - Brew lovers: Able Baker Brewing or Nevada Brew Works (casual eats + flights).
- 2:05–3:15 Arts District stroll and coffee
  - Vesta Coffee Roasters (excellent pour-overs; grab beans as a souvenir).
  - Browse local galleries and antique shops (Main St.).
- 3:30–4:30 Optional: The Neon Museum (daytime) for history; or save for evening when signs are lit. Limited shade—book timed entry.
  Indoor alternative: AREA15 (if you didn’t go Day 1) or the Downtown Container Park’s shops (semi-open but mostly covered).

Transit: Downtown ↔ Arts District is a 5-minute rideshare or 15–20 min walk. Neon Museum is 5–7 min rideshare from Fremont.

Evening (5:30–late)
Option A – Neon night + Chinatown eats
- 6:45–8:00 Neon Museum at dusk/night so the signs glow (book the Boneyard; 60–75 min).
- 8:20–10:00 Dinner in Chinatown (Spring Mountain Rd., 10–15 min rideshare)
  - Raku (reservations essential): charcoal-grilled skewers, agedashi tofu; omakase is superb.
  - Monta Ramen: rich tonkotsu; expect a short wait.
  - Chubby Cattle: hot pot with conveyor-belt fun.
- 10:00–10:45 Dessert
  - Sweets Raku (elevated plated desserts).
  - Matcha Cafe Maiko or Tiger Sugar (boba).
  - Creamberry (quirky rolled ice cream/ube treats).

Option B – All-indoor play (great if raining)
- 6:30–9:30 AREA15: Omega Mart by Meow Wolf (60–90 min), plus Illuminarium or Wink World. Casual dinner at The Beast by Todd English inside.
- 9:45–10:15 Drive by The Sphere to see the exterior animations (free viewing from pedestrian bridges near Venetian if rain eases).

Late-night add-ons (if you have energy)
- The STRAT SkyPod for 360° city lights; thrill rides for the brave.
- The Underground speakeasy at the Mob Museum (craft cocktails in a Prohibition-era setting; check hours).

Rainy-day swaps for Day 2
- Prioritize Mob Museum, Area15, and Arts District galleries/coffee. Save Neon Museum (mostly outdoors) for a dry window or swap to Pinball Hall of Fame.
- Fremont Street Experience canopy is covered but still can splash; duck into Circa or the Golden Nugget as needed.

Practical movement summary
- Strip ↔ Downtown: Deuce bus or rideshare; Monorail doesn’t go Downtown.
- Strip hopping: Monorail + free trams reduce walking, especially in heat or rain.
- Chinatown/Arts District: Short, cheap rideshare from either Strip or Downtown; sidewalks are safe and well lit.

Must-try bite checklist
- xiao long bao at Din Tai Fung
- Cronut or eclair at Dominique Ansel (Caesars)
- Secret Pizza slice (Cosmopolitan)
- Pasta + bread service at Esther’s Kitchen (Arts District)
- Robata skewers and tofu at Raku (Chinatown)
- Salted caramel or seasonal gelato at Eataly

Optional quick photo stop
- Welcome to Fabulous Las Vegas sign (15–20 min). Go early morning to avoid the queue. It’s a 15-minute walk from Mandalay Bay or a short rideshare. If raining, bring a compact umbrella; there’s no cover. Enjoy your trip!
```

### Metrics

* Model: GPT-5
* 1 Request
* 124 input tokens
* 6271k output tokens
* COST: $0.06 USD