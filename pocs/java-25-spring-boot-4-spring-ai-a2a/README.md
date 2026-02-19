# Spring AI - A2A

https://spring.io/blog/2026/01/29/spring-ai-agentic-patterns-a2a-integration

Overall <br/>

<img src="spring-ai-a2a.png" width="600" alt="Spring AI A2A">

How it works? <br/>

<img src="spring-ai-a2a-work.png" width="600" alt="Spring AI A2A Work">


## Result

Run All Agents (spring Boot)
```
❯ ./run.sh
[INFO] Scanning for projects...
[INFO] ------------------------------------------------------------------------
[INFO] Reactor Build Order:
[INFO]
[INFO] java-25-spring-boot-4-spring-ai-a2a-parent                         [pom]
[INFO] weather-agent                                                      [jar]
[INFO] hotel-agent                                                        [jar]
[INFO] trip-planner                                                       [jar]
[INFO]
[INFO] --< com.github.diegopacheco.javapocs:java-25-spring-boot-4-spring-ai-a2a-parent >--
[INFO] Building java-25-spring-boot-4-spring-ai-a2a-parent 1.0-SNAPSHOT   [1/4]
[INFO]   from pom.xml
[INFO] --------------------------------[ pom ]---------------------------------
[INFO]
[INFO] --- clean:3.5.0:clean (default-clean) @ java-25-spring-boot-4-spring-ai-a2a-parent ---
[INFO]
[INFO] -----------< com.github.diegopacheco.javapocs:weather-agent >-----------
[INFO] Building weather-agent 1.0-SNAPSHOT                                [2/4]
[INFO]   from weather-agent/pom.xml
[INFO] --------------------------------[ jar ]---------------------------------
[INFO]
[INFO] --- clean:3.5.0:clean (default-clean) @ weather-agent ---
[INFO] Deleting /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/java-25-spring-boot-4-spring-ai-a2a/weather-agent/target
[INFO]
[INFO] --- resources:3.3.1:resources (default-resources) @ weather-agent ---
[INFO] Copying 1 resource from src/main/resources to target/classes
[INFO] Copying 0 resource from src/main/resources to target/classes
[INFO]
[INFO] --- compiler:3.14.1:compile (default-compile) @ weather-agent ---
[INFO] Recompiling the module because of changed source code.
[INFO] Compiling 3 source files with javac [debug parameters release 25] to target/classes
[INFO]
[INFO] --- resources:3.3.1:testResources (default-testResources) @ weather-agent ---
[INFO] skip non existing resourceDirectory /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/java-25-spring-boot-4-spring-ai-a2a/weather-agent/src/test/resources                                                                                               [INFO]
[INFO] --- compiler:3.14.1:testCompile (default-testCompile) @ weather-agent ---
[INFO] No sources to compile
[INFO]
[INFO] --- surefire:3.5.4:test (default-test) @ weather-agent ---
[INFO] Tests are skipped.
[INFO]
[INFO] --- jar:3.4.2:jar (default-jar) @ weather-agent ---
[INFO] Building jar: /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/java-25-spring-boot-4-spring-ai-a2a/weather-agent/target/weather-agent-1.0-SNAPSHOT.jar
[INFO]
[INFO] --- spring-boot:4.0.2:repackage (repackage) @ weather-agent ---
[INFO] Replacing main artifact /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/java-25-spring-boot-4-spring-ai-a2a/weather-agent/target/weather-agent-1.0-SNAPSHOT.jar with repackaged archive, adding nested dependencies in BOOT-INF/.
[INFO] The original artifact has been renamed to /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/java-25-spring-boot-4-spring-ai-a2a/weather-agent/target/weather-agent-1.0-SNAPSHOT.jar.original
[INFO]
[INFO] ------------< com.github.diegopacheco.javapocs:hotel-agent >------------
[INFO] Building hotel-agent 1.0-SNAPSHOT                                  [3/4]
[INFO]   from hotel-agent/pom.xml
[INFO] --------------------------------[ jar ]---------------------------------
[INFO]
[INFO] --- clean:3.5.0:clean (default-clean) @ hotel-agent ---
[INFO] Deleting /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/java-25-spring-boot-4-spring-ai-a2a/hotel-agent/target
[INFO]
[INFO] --- resources:3.3.1:resources (default-resources) @ hotel-agent ---
[INFO] Copying 1 resource from src/main/resources to target/classes
[INFO] Copying 0 resource from src/main/resources to target/classes
[INFO]
[INFO] --- compiler:3.14.1:compile (default-compile) @ hotel-agent ---
[INFO] Recompiling the module because of changed source code.
[INFO] Compiling 3 source files with javac [debug parameters release 25] to target/classes
[INFO]
[INFO] --- resources:3.3.1:testResources (default-testResources) @ hotel-agent ---
[INFO] skip non existing resourceDirectory /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/java-25-spring-boot-4-spring-ai-a2a/hotel-agent/src/test/resources
[INFO]
[INFO] --- compiler:3.14.1:testCompile (default-testCompile) @ hotel-agent ---
[INFO] No sources to compile
[INFO]
[INFO] --- surefire:3.5.4:test (default-test) @ hotel-agent ---
[INFO] Tests are skipped.
[INFO]
[INFO] --- jar:3.4.2:jar (default-jar) @ hotel-agent ---
[INFO] Building jar: /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/java-25-spring-boot-4-spring-ai-a2a/hotel-agent/target/hotel-agent-1.0-SNAPSHOT.jar
[INFO]
[INFO] --- spring-boot:4.0.2:repackage (repackage) @ hotel-agent ---
[INFO] Replacing main artifact /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/java-25-spring-boot-4-spring-ai-a2a/hotel-agent/target/hotel-agent-1.0-SNAPSHOT.jar with repackaged archive, adding nested dependencies in BOOT-INF/.
[INFO] The original artifact has been renamed to /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/java-25-spring-boot-4-spring-ai-a2a/hotel-agent/target/hotel-agent-1.0-SNAPSHOT.jar.original
[INFO]
[INFO] -----------< com.github.diegopacheco.javapocs:trip-planner >------------
[INFO] Building trip-planner 1.0-SNAPSHOT                                 [4/4]
[INFO]   from trip-planner/pom.xml
[INFO] --------------------------------[ jar ]---------------------------------
[INFO]
[INFO] --- clean:3.5.0:clean (default-clean) @ trip-planner ---
[INFO] Deleting /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/java-25-spring-boot-4-spring-ai-a2a/trip-planner/target
[INFO]
[INFO] --- resources:3.3.1:resources (default-resources) @ trip-planner ---
[INFO] Copying 1 resource from src/main/resources to target/classes
[INFO] Copying 0 resource from src/main/resources to target/classes
[INFO]
[INFO] --- compiler:3.14.1:compile (default-compile) @ trip-planner ---
[INFO] Recompiling the module because of changed source code.
[INFO] Compiling 5 source files with javac [debug parameters release 25] to target/classes
[INFO]
[INFO] --- resources:3.3.1:testResources (default-testResources) @ trip-planner ---
[INFO] skip non existing resourceDirectory /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/java-25-spring-boot-4-spring-ai-a2a/trip-planner/src/test/resources
[INFO]
[INFO] --- compiler:3.14.1:testCompile (default-testCompile) @ trip-planner ---
[INFO] No sources to compile
[INFO]
[INFO] --- surefire:3.5.4:test (default-test) @ trip-planner ---
[INFO] Tests are skipped.
[INFO]
[INFO] --- jar:3.4.2:jar (default-jar) @ trip-planner ---
[INFO] Building jar: /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/java-25-spring-boot-4-spring-ai-a2a/trip-planner/target/trip-planner-1.0-SNAPSHOT.jar
[INFO]
[INFO] --- spring-boot:4.0.2:repackage (repackage) @ trip-planner ---
[INFO] Replacing main artifact /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/java-25-spring-boot-4-spring-ai-a2a/trip-planner/target/trip-planner-1.0-SNAPSHOT.jar with repackaged archive, adding nested dependencies in BOOT-INF/.
[INFO] The original artifact has been renamed to /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/java-25-spring-boot-4-spring-ai-a2a/trip-planner/target/trip-planner-1.0-SNAPSHOT.jar.original
[INFO] ------------------------------------------------------------------------
[INFO] Reactor Summary for java-25-spring-boot-4-spring-ai-a2a-parent 1.0-SNAPSHOT:
[INFO]
[INFO] java-25-spring-boot-4-spring-ai-a2a-parent ......... SUCCESS [  0.053 s]
[INFO] weather-agent ...................................... SUCCESS [  1.117 s]
[INFO] hotel-agent ........................................ SUCCESS [  0.265 s]
[INFO] trip-planner ....................................... SUCCESS [  0.296 s]
[INFO] ------------------------------------------------------------------------
[INFO] BUILD SUCCESS
[INFO] ------------------------------------------------------------------------
[INFO] Total time:  1.894 s
[INFO] Finished at: 2026-02-18T22:18:41-08:00
[INFO] ------------------------------------------------------------------------
Weather agent running on port 10001
Hotel agent running on port 10002
Trip planner running on port 10000
All A2A agents are running
```

Result:
```
❯ ./test.sh
=== Weather Agent Card ===
{"name":"Weather Agent","description":"Provides current weather conditions and forecasts for any location worldwide","url":"http://localhost:10001/a2a/","version":"1.0.0","capabilities":{"streaming":false,"pushNotifications":false,"stateTransitionHistory":false},"defaultInputModes":["text"],"defaultOutputModes":["text"],"skills":[{"id":"current_weather","name":"Current Weather","description":"Get current weather conditions for any location","tags":["weather","current","temperature"]},{"id":"weather_forecast","name":"Weather Forecast","description":"Get multi-day weather forecast for any location","tags":["weather","forecast"]}],"supportsAuthenticatedExtendedCard":false,"additionalInterfaces":[{"transport":"JSONRPC","url":"http://localhost:10001/a2a/"}],"preferredTransport":"JSONRPC","protocolVersion":"0.3.0"}
=== Hotel Agent Card ===
{"name":"Hotel Agent","description":"Provides hotel search, recommendations and availability for travel destinations","url":"http://localhost:10002/a2a/","version":"1.0.0","capabilities":{"streaming":false,"pushNotifications":false,"stateTransitionHistory":false},"defaultInputModes":["text"],"defaultOutputModes":["text"],"skills":[{"id":"hotel_search","name":"Hotel Search","description":"Search for hotels by destination and budget tier","tags":["hotel","search","accommodation"]},{"id":"hotel_availability","name":"Hotel Availability","description":"Check hotel availability and pricing for specific dates","tags":["hotel","availability","booking"]}],"supportsAuthenticatedExtendedCard":false,"additionalInterfaces":[{"transport":"JSONRPC","url":"http://localhost:10002/a2a/"}],"preferredTransport":"JSONRPC","protocolVersion":"0.3.0"}
=== Trip Planner Agent Card ===
{"name":"Trip Planner Agent","description":"Orchestrates weather and hotel agents via A2A protocol to create comprehensive trip plans","url":"http://localhost:10000/a2a/","version":"1.0.0","capabilities":{"streaming":false,"pushNotifications":false,"stateTransitionHistory":false},"defaultInputModes":["text"],"defaultOutputModes":["text"],"skills":[{"id":"trip_planning","name":"Trip Planning","description":"Plan complete trips with weather forecasts and hotel recommendations via agent orchestration","tags":["trip","travel","planning","orchestration","a2a"]}],"supportsAuthenticatedExtendedCard":false,"additionalInterfaces":[{"transport":"JSONRPC","url":"http://localhost:10000/a2a/"}],"preferredTransport":"JSONRPC","protocolVersion":"0.3.0"}
=== A2A call: Weather in Paris ===
{"jsonrpc":"2.0","id":"1","result":{"id":"c5364bdb-deec-4121-bb9c-833bec23017d","contextId":"1219eb57-3d47-49cc-803c-aec8d349fcfd","status":{"state":"completed","timestamp":"2026-02-19T06:38:55.076256Z"},"artifacts":[{"artifactId":"6833e603-a3c1-4334-9d79-4e79de00b2f8","parts":[{"text":"Current conditions (Paris, France — right now)\n- Sunny, 22 °C\n- Humidity: 60%\n- Wind: SW at 10 km/h (light breeze)\n- UV index: 5 (moderate — use sunscreen if you’ll be outside)\n\n7‑day forecast (Day 1 = today)\n- Day 1 (today): Sunny, high ~24 °C\n- Day 2 (tomorrow): Partly cloudy, high ~21 °C\n- Day 3: Rainy, high ~17 °C — bring an umbrella\n- Days 4–7: Mixed sun and clouds, highs between ~18–23 °C\n\nQuick notes and tips\n- Expect mild temperatures overall; evenings may be cooler — a light jacket is recommended.\n- Moderate UV (index 5) — sunscreen and sunglasses advisable during peak sun.\n- Rain on Day 3 could cause brief travel delays; plan accordingly.\n\nWould you like hourly forecasts, precipitation probabilities, sunrise/sunset times, or weather alerts for Paris?","kind":"text"}]}],"history":[{"role":"user","parts":[{"text":"What is the current weather and 7-day forecast for Paris, France?","kind":"text"}],"messageId":"msg-1","contextId":"1219eb57-3d47-49cc-803c-aec8d349fcfd","taskId":"c5364bdb-deec-4121-bb9c-833bec23017d","kind":"message"}],"kind":"task"}}
=== A2A call: Hotels in Rome ===
{"jsonrpc":"2.0","id":"2","result":{"id":"365d1c43-55d4-40d6-aba0-f89199a24f17","contextId":"9610792a-f4d7-4b2a-9a0a-81d79e73123c","status":{"state":"completed","timestamp":"2026-02-19T06:39:07.48469Z"},"artifacts":[{"artifactId":"3bc1f40d-491a-4667-bb47-d495cf2ef0e8","parts":[{"text":"Here are three mid-range options in Rome I found, with quick notes to help you choose. Tell me your travel dates and any preferences (neighborhood, breakfast, kitchen, near train/metro, number of guests) and I can check availability and amenities.\n\n1) Hotel Comfort Plus — ~€120/night (4-star)\n- Best for: leisure travelers wanting a comfortable, slightly upscale stay.\n- Likely features: private bathroom, daily housekeeping, on-site reception, small fitness or lounge area.\n- Location: typically centrally located — easy access to major sights by foot or short transit ride.\n- Pros: higher star rating and more hotel services; good balance of comfort and price.\n- Cons: may charge extra for breakfast or services.\n\n2) Business Traveler Inn — ~€90/night\n- Best for: budget-conscious business travelers or short stays.\n- Likely features: free breakfast, work desk(s), reliable Wi‑Fi.\n- Location: likely near business districts or transport hubs (convenient for meetings and terminals).\n- Pros: lowest nightly rate on the list and includes breakfast.\n- Cons: simpler rooms, fewer leisure amenities.\n\n3) City Center Suites — ~€110/night\n- Best for: families or longer stays who want self-catering (kitchen).\n- Likely features: kitchenette or full kitchen, apartment-style layouts, living area.\n- Location: central (good for sightseeing and markets).\n- Pros: ability to cook/save on meals and more space.\n- Cons: fewer hotel services (may be more like aparthotel).\n\nQuestions / next steps\n- Do you have travel dates? (I can check availability and exact nightly totals.)\n- How many people/rooms?\n- Any must-have amenities (breakfast included, kitchen, near Colosseum/Vatican, elevator, accessible room)?\n\nWhich of these would you like me to check availability or amenities for first?","kind":"text"}]}],"history":[{"role":"user","parts":[{"text":"Find mid-range hotels in Rome, Italy","kind":"text"}],"messageId":"msg-1","contextId":"9610792a-f4d7-4b2a-9a0a-81d79e73123c","taskId":"365d1c43-55d4-40d6-aba0-f89199a24f17","kind":"message"}],"kind":"task"}}
=== A2A call: Trip Planner orchestrates weather + hotel agents ===
{"jsonrpc":"2.0","id":"3","result":{"id":"93a5a65d-79f0-42c3-a991-8bda24b1ca5a","contextId":"f45e780d-5f05-402a-ac9e-040bcc655de2","status":{"state":"working","timestamp":"2026-02-19T06:39:07.561315Z"},"artifacts":[],"history":[{"role":"user","parts":[{"text":"Plan a 5-day trip to Barcelona, Spain with mid-range budget","kind":"text"}],"messageId":"msg-1","contextId":"f45e780d-5f05-402a-ac9e-040bcc655de2","taskId":"93a5a65d-79f0-42c3-a991-8bda24b1ca5a","kind":"message"}],"kind":"task"}}
=== REST call: Trip Planner direct endpoint ===
I can plan this — I tried to fetch live weather and hotel availability, but the travel-agent services are temporarily unreachable. I can either (A) retry the live lookups now once you give trip details, or (B) build a detailed provisional 7‑day plan and budget based on typical mid‑range prices and seasonal norms (I’ll later replace hotel listings and exact forecast with live data when the agents are reachable). Which do you prefer?
To run accurate hotel searches and a precise weather forecast I need these details:
1. Check‑in and check‑out dates (YYYY‑MM‑DD).                                                                                      2. Number of guests and number of rooms.
3. Neighborhood preference (Shinjuku, Shibuya, Ginza, Asakusa, Ueno, Tokyo Station area, “anywhere central”, or other).
4. Must‑have amenities (free breakfast, Wi‑Fi, gym, laundry, family room, accessibility).
5. Flexible on dates? If yes, how many days either side.
6. Preferred currency for prices (JPY, USD, EUR, etc.).

Below is a ready-to-use provisional plan and specific, practical advice you can accept now — I’ll swap in exact weather and hotel options once you confirm dates and let me retry the agent calls.

Provisional 7‑day Tokyo trip (mid‑range) — overview                                                                                - Accommodation suggestion: 1 mid‑range hotel in a central area (Shinjuku or Tokyo Station area recommended for transit). Typical mid‑range price: about ¥10,000–¥25,000 per room/night (~$70–$180), depending on season and neighborhood.
- Transport: get a Suica or Pasmo prepaid IC card on arrival. Expect ¥800–¥2,000/day for subway/trains depending on itinerary. Consider a JR Pass only if you plan multiple long day trips beyond Tokyo.                                                              - Food: mid‑range meals ~¥1,000–¥3,500 each. Budget ¥3,000–¥8,000/day per person for a mix of casual and one nicer dinner.
- Estimated trip cost (excluding flights): for 7 nights, mid‑range hotel + meals + local transport + attractions ≈ ¥150,000–¥350,000 per person (wide range due to hotel choice and activities). I’ll compute precise totals once you give dates and guest count.

Sample day‑by‑day itinerary (flexible)
Day 1 — Arrival + Shinjuku                                                                                                         - Arrive, check in. Pick up Suica/Pasmo.
- Explore Shinjuku: east exit (shopping), Omoide Yokocho for yakitori, Tokyo Metropolitan Government Building observation deck (free) for skyline views.
- Evening: Golden Gai or Kabukicho nightlife.

Day 2 — Shibuya & Harajuku                                                                                                         - Morning: Shibuya Crossing, Hachiko statue.
- Harajuku: Takeshita Street (youth fashion), Omotesandō (designer shops).
- Meiji Jingu shrine (calm oasis).
- Evening: Shibuya nightlife or dinner at an izakaya.

Day 3 — Asakusa + Ueno
- Senso‑ji temple and Nakamise shopping street in Asakusa.                                                                         - Sumida River walk; optional Tokyo Skytree (book tickets in advance).
- Afternoon in Ueno Park: museums (Tokyo National Museum, Ueno Zoo) or Ameya‑Yokochō market.
                                                                                                                                   Day 4 — Tsukiji outer market + Ginza + Imperial Palace
- Morning: Tsukiji outer market for fresh sushi/breakfast.
- Walk to Ginza for shopping, tea, department stores.
- Imperial Palace East Gardens (good short walk).
- Evening: Kabuki-za theatre (watch part of a performance if interested).

Day 5 — Day trip option (choose one)
- Hakone: hot springs, outdoor museum, views of Mt. Fuji (best on clear days).
- Nikko: UNESCO shrines, nature and history.
- Kamakura: Great Buddha, temples, beachside town.
I can recommend/ticket the best day trip once I know dates.

Day 6 — Odaiba and teamLab or Roppongi
- Odaiba: teamLab Planets or Borderless (book in advance), shopping malls, showy Rainbow Bridge at sunset.
- Alternative: Roppongi for Mori Art Museum and city night views.

Day 7 — Last minute shopping + departure
- Morning: pick up souvenirs in Nakamise, Ginza, or Shinjuku.
- Depending on flight time, short visit to a café or final stroll.
- Head to airport (allow 90–120 minutes by train/taxi to Narita or Haneda).

Practical tips & logistics
- Airport: Haneda is closer; Narita has more international flights. Use Keisei Skyliner (Narita) or Tokyo Monorail (Haneda) for fast access.
- Luggage: coin lockers common but not for large suitcases; consider hotel luggage storage or a takuhaibin (luggage delivery) service.
- Language: English signage common in Tokyo; learn basic phrases or use a translation app. Google Maps and Hyperdia/Google Transit work well.
- Money: credit cards widely accepted but carry some cash for smaller shops/markets. ATMs at 7‑Eleven and post offices accept international cards.
- Etiquette: be quiet on trains, queue for trains and shops, no eating while walking generally.
- Connectivity: buy a pocket Wi‑Fi or local eSIM for data.
- Safety: Tokyo is very safe; still watch belongings in busy areas.
                                                                                                                                   Packing checklist (adjust by season)                                                                                               - Comfortable walking shoes, lightweight layers, a warm coat for winter (Dec–Feb), light rain jacket for rainy season (June), umbrella.                                                                                                                               - Universal adapter, power bank, copies of important documents.
Sample mid‑range hotels & neighborhoods (provisional picks)
- Shinjuku: Hotel Gracery Shinjuku, Tokyu Stay Shinjuku, Shinjuku Granbell (good mix of convenience and value).                    - Shibuya: Cerulean Tower Tokyu Hotel (upper mid), Shibuya Excel Hotel Tokyu.                                                      - Tokyo Station / Marunouchi: Hotel Ryumeikan Tokyo, Mitsui Garden Hotel Kyobashi — best for train connections and business travel.- Asakusa / Ueno: Richmond Hotel Premier Asakusa, Mitsui Garden Hotel Ueno — more traditional/neighborhood feel.                   (When live hotel agent is available I’ll return 3–5 real‑time options with nightly rates, total cost for your exact dates, availability, and amenity filters.)

What I need from you to proceed with live data
- Provide check‑in/check‑out dates, number of guests/rooms, neighborhood preference, amenity needs, date flexibility, and currency.
- Tell me if you want me to retry the live weather + hotel searches now.

If you’re fine with a provisional plan, I can finalize a day‑by‑day schedule with restaurant suggestions and exact transit directions now. If you want live hotel availability and a current weather forecast for your trip dates, give the details above and I’ll re-run the agents. Which would you like?                                                             Done
```