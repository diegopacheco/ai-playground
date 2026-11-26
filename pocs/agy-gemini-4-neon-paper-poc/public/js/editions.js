export const EDITIONS = [
  {
    id: 'chronicle-1924',
    label: '1924 Chronicle',
    masthead: 'THE CHRONICLE GAZETTE',
    motto: 'ALL THE DISPATCHES FIT TO PRESS',
    city: 'NEW YORK & LONDON',
    date: 'TUESDAY, NOVEMBER 25, 1924',
    issue: 'VOL. LXXIV • NO. 14,892',
    price: 'TWO CENTS',
    theme: {
      paper: '#efe4cc',
      paperDark: '#dfcfa9',
      ink: '#1d1712',
      inkMuted: '#4a3d2f',
      accent: '#8b261d',
      rule: '#2c231a'
    },
    pages: [
      {
        number: 1,
        section: 'FRONT PAGE',
        runningHead: 'THE CHRONICLE GAZETTE • EVENING BROADSHEET',
        banner: 'TRANSATLANTIC AIRSHIP ARRIVES AT DAWN',
        deck: 'Dirigible Leviathan Completes Sixty-Hour Crossing Through Atlantic Gale Without Losing a Single Mail Sack',
        editionStamp: 'FINAL EDITION',
        illustration: {
          kind: 'airship',
          caption: 'FIG. 1 — The Leviathan mooring over the Hudson pier at first light, sketched by our harbor correspondent.',
          box: { x: 0.34, y: 0.30, w: 0.61, h: 0.31 }
        },
        articles: [
          {
            id: 'p1-lead',
            kicker: 'SPECIAL CABLE',
            headline: 'SILVERY HULL BREAKS THROUGH HARBOR MIST',
            byline: 'By Arthur Pendelton',
            box: { x: 0.05, y: 0.30, w: 0.27, h: 0.63 },
            lead: true,
            paragraphs: [
              'HARBOR PIER 14 — Long before the milk wagons rattled across cobbles, five thousand spectators lined the timber wharves with collars upturned against the November spray, peering into a pewter sky for the running lamps of the airship Leviathan.',
              'At twelve minutes past six, the silver envelope emerged from low fog like a quiet whale surfacing between clouds. Brass telegraphs along the pier clattered instantaneously as mooring crews hauled the forward hawser home.',
              'Captain Heinrich Voss stepped onto the gantry carrying a lacquered dispatch chest stamped in Lisbon, Paris, and Southampton, declaring the northern jet stream smoother than any railway carriage on earth.',
              'City aldermen announced a permanent steel mooring mast atop the municipal exchange before spring, promising forty-eight-hour post between continents.'
            ]
          },
          {
            id: 'p1-wireless',
            kicker: 'WIRELESS DESK',
            headline: 'MIDNIGHT SYMPHONY HEARD BY SHIPS AT SEA',
            byline: 'By Clara Vance',
            box: { x: 0.34, y: 0.63, w: 0.30, h: 0.30 },
            paragraphs: [
              'Operators aboard three North Atlantic liners logged crystal-clear violin passages transmitted from the Midtown tower at midnight.',
              'Engineers credited a newly wound copper aerial and twin vacuum tubes cooled in mineral oil for carrying the concert nine hundred miles beyond the coast.'
            ]
          },
          {
            id: 'p1-metro',
            kicker: 'CITY LEDGER',
            headline: 'PNEUMATIC MAIL TUBES EXTENDED TO HARBOR',
            byline: 'Staff Correspondent',
            box: { x: 0.66, y: 0.63, w: 0.29, h: 0.30 },
            paragraphs: [
              'Brass canisters travelling at thirty miles an hour beneath Broadway began carrying customs manifests directly to the main post room this morning.',
              'Clerks reported a fourfold reduction in horse-cart congestion along lower Fulton Street.'
            ]
          }
        ]
      },
      {
        number: 2,
        section: 'WORLD TELEGRAPH',
        runningHead: 'PAGE TWO • FOREIGN CABLES & DIPLOMACY',
        banner: 'ALPINE TUNNEL CREWS MEET UNDER GRANITE',
        deck: 'Surveyors Celebrate Three-Millimeter Alignment After Seven Years of Boring Beneath the Pass',
        illustration: {
          kind: 'locomotive',
          caption: 'FIG. 2 — Express steam locomotive testing the northern stone portal ahead of winter schedules.',
          box: { x: 0.05, y: 0.25, w: 0.56, h: 0.32 }
        },
        articles: [
          {
            id: 'p2-tunnel',
            kicker: 'GENEVA WIRE',
            headline: 'LANTERNS RAISED AS LAST ROCK WALL FALLS',
            byline: 'By Henri Laurent',
            box: { x: 0.05, y: 0.59, w: 0.56, h: 0.34 },
            paragraphs: [
              'BRIG — When the final charge cleared at dusk, northern and southern timber crews shook hands across a narrow ledge two thousand feet beneath glacial snowfields.',
              'Chief surveyor Matteo Conti checked the brass transit and confirmed the two shafts met within the width of a pocket coin, slicing nine hours off the winter mail route to Milan.',
              'Electric traction locomotives painted in deep emerald will begin hauling sleepers through the ventilated gallery on the first of December.'
            ]
          },
          {
            id: 'p2-cairo',
            kicker: 'ORIENTAL DESK',
            headline: 'DESERT CATALOGUE REVEALS BRONZE ASTROLABE',
            byline: 'By Beatrice Holt',
            box: { x: 0.64, y: 0.25, w: 0.31, h: 0.68 },
            paragraphs: [
              'ALEXANDRIA — Scholars sorting vaulted library crates uncovered an engraved bronze astrolabe bearing star tables for forty-two Mediterranean ports.',
              'Its rotating rete plate remains free of corrosion, allowing astronomers to set horizon angles with astonishing smoothness.',
              'Curators plan to display the instrument beside handwritten navigation logbooks from the sixteenth century before the winter lecture series.'
            ]
          }
        ]
      },
      {
        number: 3,
        section: 'ENGINEERING & RAIL',
        runningHead: 'PAGE THREE • MECHANICS, BRIDGES & PRESSES',
        banner: 'NEW ROTARY PRESS PRINTS FORTY THOUSAND HOURLY',
        deck: 'Twin Cast-Iron Cylinders Feed Continuous Rag-Paper Webs With Automatic Folding Knives',
        illustration: {
          kind: 'press',
          caption: 'FIG. 3 — Cross-section of the double-deck rotary cylinder press installed in Printing House Square.',
          box: { x: 0.38, y: 0.25, w: 0.57, h: 0.34 }
        },
        articles: [
          {
            id: 'p3-press',
            kicker: 'PRINTING HOUSE',
            headline: 'HOT LEAD TO FOLDED SHEET IN SIX SECONDS',
            byline: 'By Samuel Finch',
            box: { x: 0.05, y: 0.25, w: 0.30, h: 0.68 },
            paragraphs: [
              'Down in the cedar-scented basement of the Gazette building, twelve tons of machined steel began humming at four o clock this afternoon.',
              'Curved stereotype plates cast from linotype slugs lock onto revolving drums, transferring dense carbon ink onto three-mile reels of newsprint without smudging a single hairline border.',
              'Apprentices watched folded broadsheets march up brass conveyor fingers in neat stacks of fifty, ready for the street hawkers before the ink fully cooled.'
            ]
          },
          {
            id: 'p3-bridge',
            kicker: 'CIVIC WORKS',
            headline: 'SUSPENSION CABLES SPUN STRAND BY STRAND',
            byline: 'By Thomas Greer',
            box: { x: 0.38, y: 0.61, w: 0.57, h: 0.32 },
            paragraphs: [
              'Ironworkers perched three hundred feet above the East River completed the ninth galvanized wire bundle before sunset, tightening each strand to identical musical pitch.',
              'Inspectors tap the taut steel with brass mallets, listening for uniform resonance before clamping the vertical suspender ropes.'
            ]
          }
        ]
      },
      {
        number: 4,
        section: 'SCIENCE & OBSERVATORY',
        runningHead: 'PAGE FOUR • ASTRONOMY & NATURAL PHILOSOPHY',
        banner: 'GREAT REFRACTOR MAPS SPIRAL NEBULA',
        deck: 'Six-Hour Glass-Plate Exposure Reveals Millions of Faint Suns Beyond Our Milky Way',
        illustration: {
          kind: 'observatory',
          caption: 'FIG. 4 — The sixty-inch mountaintop observatory dome opened beneath a frost-clear midnight sky.',
          box: { x: 0.05, y: 0.25, w: 0.54, h: 0.33 }
        },
        articles: [
          {
            id: 'p4-stars',
            kicker: 'MOUNT WILSON',
            headline: 'ISLAND UNIVERSES CONFIRMED IN GLASS PLATES',
            byline: 'By Dr. Evelyn Marsh',
            box: { x: 0.62, y: 0.25, w: 0.33, h: 0.68 },
            paragraphs: [
              'PASADENA — Working through three freezing nights with clockwork tracking weights, astronomers resolved individual pulsing stars along the outer arms of Andromeda.',
              'By timing each variable star like a distant lighthouse, observers calculated a gulf nearly one million light-years wide, proving the cosmos extends far beyond our home galaxy.',
              'Fellows of the Royal Society hailed the glass plates as the boldest redrawing of the celestial atlas since Galileo.'
            ]
          },
          {
            id: 'p4-clock',
            kicker: 'HOROLOGY',
            headline: 'PENDULUM IN VACUUM KEEPS MILLISECOND TIME',
            byline: 'By Owen Pritchard',
            box: { x: 0.05, y: 0.60, w: 0.54, h: 0.33 },
            paragraphs: [
              'Sealed inside a copper cylinder pumped free of air, a free-swinging invar pendulum completed ninety days with less than one-tenth of a second drift.',
              'Maritime chronometer makers say radio time ticks linked to the clock will let captains fix longitude to within a ship length.'
            ]
          }
        ]
      },
      {
        number: 5,
        section: 'MARITIME & MARKETS',
        runningHead: 'PAGE FIVE • SHIPPING INTELLIGENCE & EXCHANGE',
        banner: 'COFFEE CLIPPERS BEAT AUTUMN TRADE WINDS',
        deck: 'Port Ledgers Show Record Grain Tonnage and Steady Copper Quotations Across Exchanges',
        illustration: {
          kind: 'chart',
          caption: 'FIG. 5 — Weekly exchange index and Atlantic barometric shipping lanes compiled at closing bell.',
          box: { x: 0.05, y: 0.25, w: 0.90, h: 0.30 }
        },
        articles: [
          {
            id: 'p5-exchange',
            kicker: 'EXCHANGE WIRE',
            headline: 'RAIL BONDS AND TELEGRAPH SHARES ADVANCE',
            byline: 'Financial Editor',
            box: { x: 0.05, y: 0.57, w: 0.43, h: 0.36 },
            paragraphs: [
              'Trading opened briskly beneath the vaulted skylights as Midwest grain receipts lifted trunk-line rail debentures two full points before noon.',
              'Transatlantic cable issues closed firm on news of expanded duplex relays between Nova Scotia and Cornwall.'
            ]
          },
          {
            id: 'p5-shipping',
            kicker: 'HARBOR LOG',
            headline: 'ARRIVALS, TIDES & STEAMER SAILINGS',
            byline: 'Port Master Desk',
            box: { x: 0.51, y: 0.57, w: 0.44, h: 0.36 },
            paragraphs: [
              'INWARD — SS Berengaria from Southampton with 1,420 bags of royal post; Freighter Aurora from Santos with 18,000 sacks of roasted beans.',
              'OUTWARD — SS Majestic sails at high water, 4:15 PM sharp. Light fog clearing near Ambrose Lightship.'
            ]
          }
        ]
      },
      {
        number: 6,
        section: 'ARTS & CROSSWORD',
        runningHead: 'PAGE SIX • THEATRE, BOOKS & PUZZLES',
        banner: 'JAZZ ORCHESTRA PACKS CARNEGIE BALCONIES',
        deck: 'Brass Section and Grand Piano Win Four Standing Encores in Winter Premiere',
        illustration: {
          kind: 'crossword',
          caption: 'FIG. 6 — Daily Gazette Crossword No. 412. Solutions published in tomorrow evening edition.',
          box: { x: 0.53, y: 0.25, w: 0.42, h: 0.42 }
        },
        articles: [
          {
            id: 'p6-music',
            kicker: 'CONCERT REVIEW',
            headline: 'CLARINET GLISSANDO ELECTRIFIES FORMAL HALL',
            byline: 'By Margot Sterling',
            box: { x: 0.05, y: 0.25, w: 0.45, h: 0.68 },
            paragraphs: [
              'From the opening wail of the clarinet sliding cleanly across two octaves, evening gowns and stiff collars forgot solemnity and tapped in crisp syncopation.',
              'The twenty-piece ensemble wove Harlem street rhythms seamlessly into symphonic strings, leaving critics scribbling superlatives across their playbills.',
              'Box-office clerks posted sold-out placards for the entire December fortnight before the musicians packed their mute brass.'
            ]
          },
          {
            id: 'p6-clues',
            kicker: 'PUZZLE CLUES',
            headline: 'ACROSS & DOWN — EVENING EDITION',
            byline: 'Puzzle Desk',
            box: { x: 0.53, y: 0.69, w: 0.42, h: 0.24 },
            paragraphs: [
              'ACROSS: 1. Iron typesetting line (4) • 5. Celestial map (5) • 9. Evening news sheet (7).',
              'DOWN: 1. Harbor lamp (6) • 2. Press ink roller (5) • 3. Telegraph dot (3).'
            ]
          }
        ]
      },
      {
        number: 7,
        section: 'CLASSIFIEDS & ALMANAC',
        runningHead: 'PAGE SEVEN • MERCANTILE NOTICES & INVENTIONS',
        banner: 'CLASSIFIED DISPATCHES & HOUSEHOLD WARES',
        deck: 'Fountain Pens, Pocket Chronometers, Typewriting Courses & Steamship Berths',
        illustration: {
          kind: 'adbox',
          caption: 'FIG. 7 — Authorized Mercantile Engraving: Sterling Brass Banker Lamps & Fountain Inks.',
          box: { x: 0.05, y: 0.25, w: 0.44, h: 0.36 }
        },
        articles: [
          {
            id: 'p7-lamps',
            kicker: 'MERCANTILE',
            headline: 'EMERALD GLASS BANKER LAMPS — $4.75',
            byline: 'Alder & Sons Hardware',
            box: { x: 0.05, y: 0.63, w: 0.44, h: 0.30 },
            paragraphs: [
              'Hand-blown cased green glass shades mounted on solid spun brass bases. Rests tired eyes during midnight bookkeeping and broadsheet reading.',
              'Inspect our showroom at 48 Cortlandt Street or dispatch postal order for same-day parcel delivery.'
            ]
          },
          {
            id: 'p7-classifieds',
            kicker: 'SMALL NOTICES',
            headline: 'WANTED, FOR HIRE & LOST PROPERTY',
            byline: 'Classified Clerk',
            box: { x: 0.52, y: 0.25, w: 0.43, h: 0.68 },
            paragraphs: [
              'WANTED — Experienced linotype operator for night shift. Steady wages and warm coffee provided. Apply Press Room Desk.',
              'FOR SALE — Walnut roll-top writing desk with brass pigeonholes and intact green baize blotter. $28 firm.',
              'LOST — Tortoiseshell magnifying loupe near Grand Central concourse. Reward of five dollars paid with gratitude.',
              'INSTRUCTION — Touch typewriting and Pitman shorthand taught in six weeks. Evening classes open Tuesday.'
            ]
          }
        ]
      },
      {
        number: 8,
        section: 'BACK PAGE • WEATHER',
        runningHead: 'PAGE EIGHT • BAROMETER & LATE WIRE',
        banner: 'CRISP AUTUMN HIGH PROMISES CLEAR SKIES',
        deck: 'Mercury Rises to 30.28 Inches • Brisk Northwest Breeze • Fair Weather Through Friday',
        illustration: {
          kind: 'barometer',
          caption: 'FIG. 8 — Metropolitan Barometric Dial and Coastal Isobar Sketch recorded at 6:00 PM.',
          box: { x: 0.05, y: 0.25, w: 0.50, h: 0.36 }
        },
        articles: [
          {
            id: 'p8-weather',
            kicker: 'METEOROLOGY',
            headline: 'FROST AT DAYBREAK, BRIGHT SUN BY NOON',
            byline: 'Signal Service Bureau',
            box: { x: 0.58, y: 0.25, w: 0.37, h: 0.68 },
            paragraphs: [
              'A broad ridge of dry Canadian air settled gently across the Atlantic seaboard this evening, sweeping out harbor mists and lifting the brass barometer needle cleanly into Fair.',
              'High tomorrow near 48 degrees; overnight low 33 degrees with sparkling stars ideal for rooftop stargazing.',
              'Coastal steamers report calm three-foot swells from Cape Cod south to Chesapeake Bay.'
            ]
          },
          {
            id: 'p8-latewire',
            kicker: 'STOP PRESS',
            headline: 'LATE TELEGRAPH BULLETINS',
            byline: 'Night Wire',
            box: { x: 0.05, y: 0.63, w: 0.50, h: 0.30 },
            paragraphs: [
              'LONDON — Fog lifts along Thames embankments; omnibus schedules return to normal intervals.',
              'CHICAGO — Elevated line unveils heated oak passenger cars ahead of winter cold snap.'
            ]
          }
        ]
      }
    ]
  },
  {
    id: 'neon-1989',
    label: '1989 Neon Observer',
    masthead: 'THE NEON OBSERVER',
    motto: 'SIGNAL, SILICON & METROPOLITAN WIRE',
    city: 'SAN FRANCISCO & TOKYO',
    date: 'SATURDAY, NOVEMBER 25, 1989',
    issue: 'VOL. XII • NO. 4,096',
    price: '50 CENTS',
    theme: {
      paper: '#eae6dc',
      paperDark: '#d5cfc2',
      ink: '#141820',
      inkMuted: '#3e4654',
      accent: '#155d82',
      rule: '#1f2836'
    },
    pages: [
      {
        number: 1,
        section: 'FRONT PAGE',
        runningHead: 'THE NEON OBSERVER • LATE METRO EDITION',
        banner: 'GLOBAL HYPERTEXT WEB PROPOSAL CIRCULATES',
        deck: 'Researchers Draft Open Document Protocol Linking Workstations Across Continents Over Fiber',
        editionStamp: 'LATE WIRE',
        illustration: {
          kind: 'chart',
          caption: 'FIG. 1 — Backbone packet routing map joining research laboratories across three time zones.',
          box: { x: 0.34, y: 0.30, w: 0.61, h: 0.31 }
        },
        articles: [
          {
            id: 'n1-lead',
            kicker: 'NETWORKS',
            headline: 'PLAIN TEXT LINKS PROMISE UNIVERSAL LIBRARY',
            byline: 'By Naomi Takahashi',
            box: { x: 0.05, y: 0.30, w: 0.27, h: 0.63 },
            lead: true,
            paragraphs: [
              'GENEVA — A slim technical memorandum circulating among physics laboratories proposes binding documents across the globe with simple clickable references.',
              'Instead of locked proprietary terminals, any workstation speaking the open hypertext wire could fetch pages from servers oceans away in fractions of a second.',
              'Early sceptics called the architecture deceptively plain, yet engineers who tested the first browser marveled at jumping effortlessly from Zurich data tables to California archives.',
              'Telecom observers predict newsstands themselves may someday glow on desktop cathode tubes.'
            ]
          },
          {
            id: 'n1-gpu',
            kicker: 'GRAPHICS',
            headline: 'REAL-TIME POLYGONS REACH DESKTOP SCREENS',
            byline: 'By Marcus Sterling',
            box: { x: 0.34, y: 0.63, w: 0.30, h: 0.30 },
            paragraphs: [
              'Dedicated geometry chips unveiled at the autumn symposium shaded forty thousand triangles per second with crisp Gouraud lighting.',
              'Designers rotated architectural blueprints fluidly with a single wrist movement on a three-button mouse.'
            ]
          },
          {
            id: 'n1-synth',
            kicker: 'AUDIO LAB',
            headline: 'FM SYNTHESIZERS RESHAPE NIGHT RADIO',
            byline: 'Staff Writer',
            box: { x: 0.66, y: 0.63, w: 0.29, h: 0.30 },
            paragraphs: [
              'Independent late-night studios traded refrigerator-sized tape racks for compact digital patch bays capable of glassy bells and warm analog brass.',
              'Listeners flooded switchboards requesting full cassette recordings of the midnight broadcast.'
            ]
          }
        ]
      },
      {
        number: 2,
        section: 'WORLD WIRE',
        runningHead: 'PAGE TWO • PACIFIC & EUROPEAN DISPATCHES',
        banner: 'SHINKANSEN TESTS MAGNETIC LEVITATION TRACK',
        deck: 'Superconducting Prototype Glides Silently Above Concrete Guideway at Record Speeds',
        illustration: {
          kind: 'locomotive',
          caption: 'FIG. 2 — Aerodynamic test car banking smoothly along the Miyazaki coastal test corridor.',
          box: { x: 0.05, y: 0.25, w: 0.56, h: 0.32 }
        },
        articles: [
          {
            id: 'n2-maglev',
            kicker: 'TOKYO BUREAU',
            headline: 'NO WHEELS, NO FRICTION, ONLY MAGNETIC CUSHION',
            byline: 'By Kenji Sato',
            box: { x: 0.05, y: 0.59, w: 0.56, h: 0.34 },
            paragraphs: [
              'MIYAZAKI — Observers watching from the concrete overpass heard only a brisk rush of wind as the sleek test coach floated four inches above its coils.',
              'Cryogenic magnets held the vehicle centered through sweeping curves while onboard telemetry streamed cleanly back to the control booth.'
            ]
          },
          {
            id: 'n2-sat',
            kicker: 'ORBITAL DESK',
            headline: 'WEATHER SATELLITE SENDS HIGH-RES TYPHOON LOOP',
            byline: 'By Elena Rostova',
            box: { x: 0.64, y: 0.25, w: 0.31, h: 0.68 },
            paragraphs: [
              'Geostationary infrared sensors began relaying cloud-top temperatures every fifteen minutes, giving maritime fleets forty-eight hours of extra warning.',
              'Forecasters stitched the monochrome frames into smooth animations showing spiral eye-walls forming over warm western Pacific currents.'
            ]
          }
        ]
      },
      {
        number: 3,
        section: 'SILICON & PRESS',
        runningHead: 'PAGE THREE • TYPESETTING & HARDWARE',
        banner: 'LASER TYPESETTERS RETIRE HOT METAL FOUNDRIES',
        deck: 'PostScript Page Descriptions Deliver Razor-Sharp Serif Typography at 2,400 Dots Per Inch',
        illustration: {
          kind: 'press',
          caption: 'FIG. 3 — Modern web-offset press marrying laser-etched plates with classic newsprint rolls.',
          box: { x: 0.38, y: 0.25, w: 0.57, h: 0.34 }
        },
        articles: [
          {
            id: 'n3-laser',
            kicker: 'COMPOSING ROOM',
            headline: 'BEZIER CURVES REPLACE LEAD MATRICES',
            byline: 'By David Lindholm',
            box: { x: 0.05, y: 0.25, w: 0.30, h: 0.68 },
            paragraphs: [
              'Veterans who spent thirty winters breathing molten lead mist stood quietly as a desktop page layout printed a complete front-page proof in forty seconds.',
              'Mathematical spline outlines scale classic nineteenth-century display fonts to any headline size without jagged stair-stepping.'
            ]
          },
          {
            id: 'n3-cd',
            kicker: 'STORAGE',
            headline: 'WHOLE ENCYCLOPEDIA FITS ON ONE SILVER DISC',
            byline: 'By Priya Nair',
            box: { x: 0.38, y: 0.61, w: 0.57, h: 0.32 },
            paragraphs: [
              'Optical mastering labs demonstrated six hundred megabytes of indexed text and line engravings pressed onto polycarbonate discs thinner than a dime.',
              'Public libraries lined up to order drives for reference reading rooms.'
            ]
          }
        ]
      },
      {
        number: 4,
        section: 'SPACE & SCIENCE',
        runningHead: 'PAGE FOUR • DEEP SPACE & OPTICS',
        banner: 'VOYAGER SENDS AZURE PORTRAITS OF NEPTUNE',
        deck: 'Twelve-Year Robotic Voyage Reveals Supersonic Methane Winds and Geysers on Triton',
        illustration: {
          kind: 'observatory',
          caption: 'FIG. 4 — Deep Space Network dish array locking onto whisper-faint telemetry from the outer solar system.',
          box: { x: 0.05, y: 0.25, w: 0.54, h: 0.33 }
        },
        articles: [
          {
            id: 'n4-voyager',
            kicker: 'PASADENA',
            headline: 'FAINT RADIO WHISPER CROSSES FOUR HOURS OF VOID',
            byline: 'By Carlotta Vega',
            box: { x: 0.62, y: 0.25, w: 0.33, h: 0.68 },
            paragraphs: [
              'Arriving with less power than a wristwatch battery, radio pulses from three billion miles out painted vivid cobalt clouds across mission monitors.',
              'Engineers applauded as the aging spacecraft swung past icy Triton before turning forever toward interstellar night.'
            ]
          },
          {
            id: 'n4-fiber',
            kicker: 'OPTICS',
            headline: 'TRANSATLANTIC GLASS CABLE CARRIES 40,000 CALLS',
            byline: 'By Henrik Lind',
            box: { x: 0.05, y: 0.60, w: 0.54, h: 0.33 },
            paragraphs: [
              'Hair-thin silica fibers resting on the abyssal seafloor carried pristine laser pulses between New Jersey and Brittany with zero echo.'
            ]
          }
        ]
      },
      {
        number: 5,
        section: 'MARKETS & TRADE',
        runningHead: 'PAGE FIVE • SEMICONDUCTOR & SHIPPING LEDGER',
        banner: 'MEMORY CHIPS & FIBER OPTICS LEAD RALLY',
        deck: 'Pacific Container Terminals Automate Manifests While Workstation Orders Double',
        illustration: {
          kind: 'chart',
          caption: 'FIG. 5 — Closing technology index and Pacific freight throughput.',
          box: { x: 0.05, y: 0.25, w: 0.90, h: 0.30 }
        },
        articles: [
          {
            id: 'n5-markets',
            kicker: 'EXCHANGE',
            headline: 'RISC WORKSTATIONS POST RECORD BACKLOG',
            byline: 'Markets Desk',
            box: { x: 0.05, y: 0.57, w: 0.43, h: 0.36 },
            paragraphs: [
              'Institutional desks bid up microprocessor and precision optics shares following strong autumn shipments across universities and engineering firms.'
            ]
          },
          {
            id: 'n5-port',
            kicker: 'FREIGHT',
            headline: 'BARCODE SCANNERS CLEAR DOCKS IN HALF TIME',
            byline: 'Shipping Desk',
            box: { x: 0.51, y: 0.57, w: 0.44, h: 0.36 },
            paragraphs: [
              'Handheld laser wands linked to pier terminals cut customs clearance on transpacific cargo from two days to four hours.'
            ]
          }
        ]
      },
      {
        number: 6,
        section: 'CULTURE & PUZZLE',
        runningHead: 'PAGE SIX • CINEMA, ARCADES & CROSSWORD',
        banner: 'VECTOR ARCADE CABINETS ENTER DESIGN MUSEUM',
        deck: 'Curators Praise Wireframe Geometry as Pure Kinetic Sculpture of the Electronic Age',
        illustration: {
          kind: 'crossword',
          caption: 'FIG. 6 — Observer Logic Grid No. 89. Solve with fountain pen or pencil.',
          box: { x: 0.53, y: 0.25, w: 0.42, h: 0.42 }
        },
        articles: [
          {
            id: 'n6-arcade',
            kicker: 'DESIGN',
            headline: 'GLOWING PHOSPHOR LINES CELEBRATED AS ART',
            byline: 'By Simone Delacroix',
            box: { x: 0.05, y: 0.25, w: 0.45, h: 0.68 },
            paragraphs: [
              'Visitors stood mesmerized in darkened gallery rooms watching luminous monochrome vectors trace starfields and rotating polyhedra without a single pixel.',
              'Curators compared the crisp glowing ink of vector monitors to classic copperplate engravings.'
            ]
          },
          {
            id: 'n6-clues',
            kicker: 'CLUES',
            headline: 'ACROSS & DOWN',
            byline: 'Grid Editor',
            box: { x: 0.53, y: 0.69, w: 0.42, h: 0.24 },
            paragraphs: [
              'ACROSS: 1. Wireframe face (4) • 5. Optical beam (5) • 9. Code module (6).',
              'DOWN: 1. Binary bit (5) • 2. Matrix turn (6) • 3. Screen glow (4).'
            ]
          }
        ]
      },
      {
        number: 7,
        section: 'CLASSIFIEDS',
        runningHead: 'PAGE SEVEN • WORKSTATIONS & PARTS',
        banner: 'NEON OBSERVER CLASSIFIED MARKET',
        deck: 'Mechanical Keyboards, Oscilloscopes, Vintage Lenses & Technical Manuals',
        illustration: {
          kind: 'adbox',
          caption: 'FIG. 7 — Precision Drafting Lamps & Buckling-Spring Terminals.',
          box: { x: 0.05, y: 0.25, w: 0.44, h: 0.36 }
        },
        articles: [
          {
            id: 'n7-ad',
            kicker: 'HARDWARE',
            headline: 'SOLID STEEL TACTILE KEYBOARDS — $89',
            byline: 'Pacific Peripherals',
            box: { x: 0.05, y: 0.63, w: 0.44, h: 0.30 },
            paragraphs: [
              'Engineered for fifty million crisp keystrokes with coiled cable and stepped keycaps. Built to outlast three generations of computers.'
            ]
          },
          {
            id: 'n7-list',
            kicker: 'NOTICES',
            headline: 'ENGINEERS, SURPLUS & SWAP MEET',
            byline: 'Classified Desk',
            box: { x: 0.52, y: 0.25, w: 0.43, h: 0.68 },
            paragraphs: [
              'SEEKING — Graphics programmer fluent in C and matrix transforms for small studio near Embarcadero.',
              'FOR SALE — Dual-trace 100MHz oscilloscope with fresh probes and printed schematic binder.',
              'MEETUP — Sunday electronics swap at Foothill lot, gates open 6:00 AM sharp.'
            ]
          }
        ]
      },
      {
        number: 8,
        section: 'WEATHER & LATE WIRE',
        runningHead: 'PAGE EIGHT • COASTAL RADAR & FORECAST',
        banner: 'CLEAR PACIFIC RIDGE SWEEPS BAY FOG',
        deck: 'Barometer Steady 30.14 • Crisp Visibility Across Golden Gate • Light Offshore Breeze',
        illustration: {
          kind: 'barometer',
          caption: 'FIG. 8 — Coastal Barometer & Wind Rose plotted at 18:00 PST.',
          box: { x: 0.05, y: 0.25, w: 0.50, h: 0.36 }
        },
        articles: [
          {
            id: 'n8-weather',
            kicker: 'FORECAST',
            headline: 'BRIGHT SUNSHINE THROUGH WEEKEND',
            byline: 'Bay Weather Bureau',
            box: { x: 0.58, y: 0.25, w: 0.37, h: 0.68 },
            paragraphs: [
              'Dry northwesterly winds brushed marine stratus miles offshore, leaving crisp sixty-degree afternoons and starry evenings across the hills.'
            ]
          },
          {
            id: 'n8-wire',
            kicker: 'LATE WIRE',
            headline: 'FINAL DISPATCHES BEFORE PRESS LOCK',
            byline: 'Night Desk',
            box: { x: 0.05, y: 0.63, w: 0.50, h: 0.30 },
            paragraphs: [
              'TOKYO — Dawn fish auction opens under cloudless skies.',
              'LONDON — Subsea tunnel boring shields pass halfway chalk stratum.'
            ]
          }
        ]
      }
    ]
  },
  {
    id: 'aether-1899',
    label: '1899 Aether Chronicle',
    masthead: 'THE AETHER CHRONICLE',
    motto: 'SCIENTIFIC VOYAGES & CELESTIAL GAZETTE',
    city: 'LONDON & PARIS',
    date: 'SATURDAY, NOVEMBER 25, 1899',
    issue: 'VOL. VII • NO. 918',
    price: 'ONE PENNY',
    theme: {
      paper: '#ebd8b3',
      paperDark: '#d6be90',
      ink: '#24180d',
      inkMuted: '#564028',
      accent: '#23594f',
      rule: '#322214'
    },
    pages: [
      {
        number: 1,
        section: 'FRONT PAGE',
        runningHead: 'THE AETHER CHRONICLE • ROYAL SCIENTIFIC BROADSHEET',
        banner: 'BRASS AUTOMATON WINS PARIS CHESS SALON',
        deck: 'Clockwork Analytical Engine Calculates Seven Moves Deep Using Punched Ivory Cards',
        editionStamp: 'ROYAL EDITION',
        illustration: {
          kind: 'airship',
          caption: 'FIG. 1 — Exhibition sky-vessel moored beside the iron tower prior to the evening lecture.',
          box: { x: 0.34, y: 0.30, w: 0.61, h: 0.31 }
        },
        articles: [
          {
            id: 'a1-lead',
            kicker: 'PARIS DISPATCH',
            headline: 'GEARS CLICK SOFTLY AS GRANDMASTER RESIGNS',
            byline: 'By Prof. Alistair Crowe',
            box: { x: 0.05, y: 0.30, w: 0.27, h: 0.63 },
            lead: true,
            paragraphs: [
              'PARIS — Beneath chandeliers burning whale-oil gas, a mahogany cabinet of interlocking brass pinions defeated three Parisian champions in succession without a single human whisper.',
              'Driven solely by descending lead counterweights and stacks of perforated ivory cards, the engine announced checkmate by ringing a tiny silver bell.',
              'Members of the Academy inspected every gear housing with lanterns to confirm no concealed dwarf sat inside, declaring the triumph of mechanical logic complete.'
            ]
          },
          {
            id: 'a1-balloon',
            kicker: 'AERONAUTICS',
            headline: 'SILK BALLOON ASCENDS ABOVE WINTER CLOUDS',
            byline: 'By Jules Mercier',
            box: { x: 0.34, y: 0.63, w: 0.30, h: 0.30 },
            paragraphs: [
              'Equipped with aneroid barometers and bottled oxygen, two aeronauts rose five miles over the Seine and sketched crystal ice halos in ink.'
            ]
          },
          {
            id: 'a1-sub',
            kicker: 'NAUTICAL',
            headline: 'ELECTRIC SUBMERSIBLE GLIDES UNDER CHANNEL',
            byline: 'Maritime Correspondent',
            box: { x: 0.66, y: 0.63, w: 0.29, h: 0.30 },
            paragraphs: [
              'Powered by banks of lead accumulators, the copper-hulled vessel cruised forty feet below Dover ferries with glowing glass viewports.'
            ]
          }
        ]
      },
      {
        number: 2,
        section: 'CONTINENTAL POST',
        runningHead: 'PAGE TWO • EXPEDITIONS & RAILWAYS',
        banner: 'ORIENT EXPRESS ADDS OBSERVATION CAR',
        deck: 'Teak Panelling, Brass Reading Lamps and Panoramic Bevelled Windows Debut on Winter Run',
        illustration: {
          kind: 'locomotive',
          caption: 'FIG. 2 — Compound express engine departing Gare de l Est under billowing steam.',
          box: { x: 0.05, y: 0.25, w: 0.56, h: 0.32 }
        },
        articles: [
          {
            id: 'a2-rail',
            kicker: 'VIENNA POST',
            headline: 'VELVET SALONS ROLL ACROSS DANUBE BRIDGES',
            byline: 'By Clara Lindqvist',
            box: { x: 0.05, y: 0.59, w: 0.56, h: 0.34 },
            paragraphs: [
              'Travellers bound for Constantinople now dine beneath brass gasoliers while watching frosted pine valleys glide past crystal panes.'
            ]
          },
          {
            id: 'a2-arctic',
            kicker: 'POLAR WIRE',
            headline: 'ICEBOUND BRIG SENDS CARRIER PIGEON LOG',
            byline: 'Royal Geographical Desk',
            box: { x: 0.64, y: 0.25, w: 0.31, h: 0.68 },
            paragraphs: [
              'A weathered tin capsule strapped to a northern homing bird reached Tromso with compass bearings and sketches of emerald aurora curtains.'
            ]
          }
        ]
      },
      {
        number: 3,
        section: 'WORKSHOPS',
        runningHead: 'PAGE THREE • FOUNDRIES & HOROLOGY',
        banner: 'STEAM PRINTING PRESS UNVEILED AT FLEET STREET',
        deck: 'Precision Brass Typefaces and Fine Linen Paper Raise Broadsheet Clarity',
        illustration: {
          kind: 'press',
          caption: 'FIG. 3 — Steam-belted printing engine inked with lampblack and linseed varnish.',
          box: { x: 0.38, y: 0.25, w: 0.57, h: 0.34 }
        },
        articles: [
          {
            id: 'a3-press',
            kicker: 'FOUNDRY',
            headline: 'ENGRAVERS CUT WOODBLOCKS BY GASLIGHT',
            byline: 'By Gideon Frost',
            box: { x: 0.05, y: 0.25, w: 0.30, h: 0.68 },
            paragraphs: [
              'Boxwood blocks seasoned for seven years yield crisp crosshatched illustrations capable of enduring fifty thousand impressions.'
            ]
          },
          {
            id: 'a3-clock',
            kicker: 'GREENWICH',
            headline: 'TIME BALL DROPS AT ONE O CLOCK SHARP',
            byline: 'Observatory Clerk',
            box: { x: 0.38, y: 0.61, w: 0.57, h: 0.32 },
            paragraphs: [
              'Captains moored along the Thames raised brass spyglasses toward Flamsteed House to synchronize shipboard chronometers.'
            ]
          }
        ]
      },
      {
        number: 4,
        section: 'CELESTIAL ATLAS',
        runningHead: 'PAGE FOUR • TELESCOPES & PLANETS',
        banner: 'MARS OPPOSITION DRAWS HUNDRED TELESCOPES',
        deck: 'Clear Autumn Night Lets Observers Sketch Polar Snowcaps in Fine Charcoal',
        illustration: {
          kind: 'observatory',
          caption: 'FIG. 4 — Equatorial brass telescope rotating smoothly on cast-iron tracks.',
          box: { x: 0.05, y: 0.25, w: 0.54, h: 0.33 }
        },
        articles: [
          {
            id: 'a4-mars',
            kicker: 'ASTRONOMY',
            headline: 'OCHRE DESERTS SHIMMER IN EYEPIECE',
            byline: 'By Dr. Lydia Shaw',
            box: { x: 0.62, y: 0.25, w: 0.33, h: 0.68 },
            paragraphs: [
              'High above city coal smoke, observers recorded crisp dust storms swirling across the ruddy southern hemisphere.'
            ]
          },
          {
            id: 'a4-lens',
            kicker: 'OPTICS',
            headline: 'FLINT GLASS BLANK COOLS OVER TWELVE MONTHS',
            byline: 'Optical Guild',
            box: { x: 0.05, y: 0.60, w: 0.54, h: 0.33 },
            paragraphs: [
              'Master polishers finished a forty-inch objective lens free of bubbles, promising unmatched views of Saturn rings.'
            ]
          }
        ]
      },
      {
        number: 5,
        section: 'EXCHANGE & PORTS',
        runningHead: 'PAGE FIVE • TEA CLIPPERS & BULLION',
        banner: 'TEA CLIPPER ARRIVES IN NINETY-ONE DAYS',
        deck: 'Full Canvas Run From Foochow Wins Coveted Dockside Premium',
        illustration: {
          kind: 'chart',
          caption: 'FIG. 5 — Trade wind velocity and commodity ledger.',
          box: { x: 0.05, y: 0.25, w: 0.90, h: 0.30 }
        },
        articles: [
          {
            id: 'a5-tea',
            kicker: 'THAMES DOCKS',
            headline: 'CHESTS OF FRESH PEKOE AUCTIONED AT DAWN',
            byline: 'Mercantile Desk',
            box: { x: 0.05, y: 0.57, w: 0.43, h: 0.36 },
            paragraphs: [
              'Brokers crowded Mincing Lane as fragrant samples from the winning clipper commanded handsome winter bids.'
            ]
          },
          {
            id: 'a5-wire',
            kicker: 'CABLES',
            headline: 'SUBMARINE TELEGRAPH RATES LOWERED',
            byline: 'Post Office Notice',
            box: { x: 0.51, y: 0.57, w: 0.44, h: 0.36 },
            paragraphs: [
              'New gutta-percha insulated cables doubled message speed between London, Bombay, and Sydney.'
            ]
          }
        ]
      },
      {
        number: 6,
        section: 'SALON & RIDDLES',
        runningHead: 'PAGE SIX • LITERATURE & ENIGMAS',
        banner: 'LANTERN SLIDE LECTURE DELIGHTS ROYAL HALL',
        deck: 'Hand-Tinted Glass Plates Project Deep-Sea Coral Gardens Twenty Feet High',
        illustration: {
          kind: 'crossword',
          caption: 'FIG. 6 — Acrostic Word Square No. 108.',
          box: { x: 0.53, y: 0.25, w: 0.42, h: 0.42 }
        },
        articles: [
          {
            id: 'a6-salon',
            kicker: 'LECTURE',
            headline: 'BIOLUMINESCENT CREATURES OF THE ABYSS',
            byline: 'By Beatrice Finch',
            box: { x: 0.05, y: 0.25, w: 0.45, h: 0.68 },
            paragraphs: [
              'Naturalists unveiled hand-painted glass transparencies depicting glowing lanternfish dredged three miles beneath the Atlantic.'
            ]
          },
          {
            id: 'a6-clues',
            kicker: 'ENIGMAS',
            headline: 'ACROSS & DOWN',
            byline: 'Riddle Master',
            box: { x: 0.53, y: 0.69, w: 0.42, h: 0.24 },
            paragraphs: [
              'ACROSS: 1. Brass compass (5) • 5. Star lens (5).',
              'DOWN: 1. Steam ship (6) • 2. Ink quill (5).'
            ]
          }
        ]
      },
      {
        number: 7,
        section: 'CURIOSITIES',
        runningHead: 'PAGE SEVEN • APOTHECARY & INSTRUMENTS',
        banner: 'ROYAL INSTRUMENT MAKERS & BOOKSELLERS',
        deck: 'Brass Sextants, Pocket Telescopes, Cedar Pencils & Leather Folios',
        illustration: {
          kind: 'adbox',
          caption: 'FIG. 7 — Whitfield & Co. Optical Loupes and Brass Compasses.',
          box: { x: 0.05, y: 0.25, w: 0.44, h: 0.36 }
        },
        articles: [
          {
            id: 'a7-loupe',
            kicker: 'OPTICAL',
            headline: 'FOLDING BRASS MAGNIFYING LOUPE — 6 SHILLINGS',
            byline: 'Whitfield & Co.',
            box: { x: 0.05, y: 0.63, w: 0.44, h: 0.30 },
            paragraphs: [
              'Triple-ground crown glass lens in solid satin brass bezel. Essential for botanists, watchmakers, and broadsheet connoisseurs.'
            ]
          },
          {
            id: 'a7-notices',
            kicker: 'POSTINGS',
            headline: 'EXPEDITIONS & APPRENTICESHIPS',
            byline: 'Registry',
            box: { x: 0.52, y: 0.25, w: 0.43, h: 0.68 },
            paragraphs: [
              'WANTED — Skilled cartographer for spring Mediterranean sounding voyage. Cabins and drafting instruments furnished.',
              'BOOKS — Complete leather atlas of 1898 offered at Charing Cross stall.'
            ]
          }
        ]
      },
      {
        number: 8,
        section: 'ALMANAC & GLASS',
        runningHead: 'PAGE EIGHT • WEATHER GLASS & TIDES',
        banner: 'RISING MERCURY HERALDS BRISK CLEAR MORNING',
        deck: 'Barometer 30.32 • High Tide London Bridge 2:40 PM • Waxing Crescent Moon',
        illustration: {
          kind: 'barometer',
          caption: 'FIG. 8 — FitzRoy Storm Glass and Barometric Reading.',
          box: { x: 0.05, y: 0.25, w: 0.50, h: 0.36 }
        },
        articles: [
          {
            id: 'a8-weather',
            kicker: 'ALMANAC',
            headline: 'CLEAR FROSTED ROOFTOPS BY SUNRISE',
            byline: 'Meteorological Office',
            box: { x: 0.58, y: 0.25, w: 0.37, h: 0.68 },
            paragraphs: [
              'Crisp easterly breezes swept river mist toward the estuary, promising dry cobbles and bright golden afternoon light.'
            ]
          },
          {
            id: 'a8-post',
            kicker: 'LATE POST',
            headline: 'EVENING TELEGRAPH NOTES',
            byline: 'Night Clerk',
            box: { x: 0.05, y: 0.63, w: 0.50, h: 0.30 },
            paragraphs: [
              'DOVER — Packet steamer crossed in seventy-eight minutes on calm swell.',
              'EDINBURGH — New observatory dome painted ivory ahead of winter solstice.'
            ]
          }
        ]
      }
    ]
  }
];

export function cloneEdition(edition) {
  return structuredClone(edition);
}

export function getEditionById(id) {
  return EDITIONS.find((e) => e.id === id) || EDITIONS[0];
}

export function applyHeadlineEdit(edition, { masthead, banner, deck, leadBody }) {
  const next = cloneEdition(edition);
  if (typeof masthead === 'string' && masthead.trim()) {
    next.masthead = masthead.trim().toUpperCase().slice(0, 40);
  }
  const front = next.pages[0];
  if (typeof banner === 'string' && banner.trim()) {
    front.banner = banner.trim().toUpperCase().slice(0, 72);
  }
  if (typeof deck === 'string' && deck.trim()) {
    front.deck = deck.trim().slice(0, 160);
  }
  if (typeof leadBody === 'string' && leadBody.trim()) {
    const lead = front.articles.find((a) => a.lead) || front.articles[0];
    if (lead) {
      lead.paragraphs = leadBody
        .split(/\n+/)
        .map((p) => p.trim())
        .filter(Boolean);
    }
  }
  return next;
}

export function hitTestArticle(page, u, v) {
  if (!page || !Array.isArray(page.articles)) return null;
  for (const article of page.articles) {
    const b = article.box;
    if (u >= b.x && u <= b.x + b.w && v >= b.y && v <= b.y + b.h) {
      return article;
    }
  }
  return null;
}

export function validateEdition(edition) {
  const errors = [];
  if (!edition || typeof edition !== 'object') {
    return ['edition must be an object'];
  }
  if (!edition.id || !edition.masthead) {
    errors.push('edition requires id and masthead');
  }
  if (!Array.isArray(edition.pages) || edition.pages.length !== 8) {
    errors.push('edition must have exactly 8 pages');
    return errors;
  }
  edition.pages.forEach((page, idx) => {
    if (page.number !== idx + 1) {
      errors.push(`page ${idx} number mismatch`);
    }
    if (!Array.isArray(page.articles) || page.articles.length === 0) {
      errors.push(`page ${page.number} has no articles`);
      return;
    }
    for (const art of page.articles) {
      const b = art.box;
      if (!b || b.x < 0 || b.y < 0 || b.w <= 0 || b.h <= 0 || b.x + b.w > 1.01 || b.y + b.h > 1.01) {
        errors.push(`article ${art.id} box out of bounds`);
      }
    }
  });
  return errors;
}
