import json
import os
import re
import time
import urllib.parse
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
COVERS = os.path.join(BASE, "static", "covers")
CONSOLES = os.path.join(BASE, "static", "consoles")
UA = {"User-Agent": "GameStand/1.0 (diego.pacheco.it@gmail.com) python-urllib"}

GAMES = [
    ("Street Fighter 5", "PS5", "Street Fighter V", 2016),
    ("Mortal Kombat X", "PS5", None, 2016),
    ("Walking Dead Season 2", "PS5", "The Walking Dead Season Two video game", 2014),
    ("South Park: The Stick of Truth", "PS5", None, 2014),
    ("Diablo 3 - Expansion 1", "PS5", "Diablo III Reaper of Souls", 2017),
    ("Batman Arkham City", "PS5", "Batman Arkham City", 2016),
    ("Batman Arkham Asylum", "PS5", "Batman Arkham Asylum", 2016),
    ("SC2 - The Legacy of the Void", "Steam", "StarCraft II Legacy of the Void", 2015),
    ("Dishonored 2", "PS5", None, 2017),
    ("FFXV", "PS5", "Final Fantasy XV", 2017),
    ("Injustice 2", "PS5", None, 2017),
    ("Marvel VS Capcom Infinite", "PS5", "Marvel vs. Capcom: Infinite", 2018),
    ("Bloodborne", "PS5", None, 2018),
    ("God of War", "PS5", "God of War 2018 video game", 2018),
    ("Spiderman", "PS5", "Marvel's Spider-Man 2018", 2018),
    ("Trine", "Steam", "Trine video game", 2020),
    ("RE2 Remake", "PS5", "Resident Evil 2 2019 video game", 2020),
    ("RE3 Remake", "PS5", "Resident Evil 3 2020 video game", 2020),
    ("FF7R", "PS5", "Final Fantasy VII Remake", 2020),
    ("Tomb Raider (2013)", "Steam", "Tomb Raider 2013 video game", 2020),
    ("Deus Ex Mankind Divided", "Steam", None, 2020),
    ("Rise of the Tomb Raider", "Steam", None, 2020),
    ("Shadow of the Tomb Raider", "Steam", None, 2020),
    ("Dying Light", "Steam", "Dying Light video game", 2020),
    ("Carrion", "Steam", "Carrion video game", 2020),
    ("Rocketbirds: Hardboiled Chicken", "Steam", None, 2020),
    ("Alien Isolation", "Steam", "Alien: Isolation", 2020),
    ("Shadow Tactics", "Steam", "Shadow Tactics: Blades of the Shogun", 2020),
    ("The Mark of The Ninja Remastered", "Steam", "Mark of the Ninja", 2020),
    ("Hitman 2016", "Steam", "Hitman 2016 video game", 2020),
    ("Desperados III", "Steam", None, 2020),
    ("CyberPunk 2077", "PS5", "Cyberpunk 2077", 2020),
    ("Factorio", "Steam", None, 2021),
    ("Last Epoch", "Steam", None, 2021),
    ("Portal Reloaded", "Steam", None, 2021),
    ("Resident Evil 8 - Village", "PS5", "Resident Evil Village", 2021),
    ("Hitman 2", "PS5", "Hitman 2 2018 video game", 2021),
    ("Hitman 3", "PS5", None, 2021),
    ("Ghost of Tsushima", "PS5", None, 2021),
    ("Mortal Kombat 11 / Aftermath", "PS5", "Mortal Kombat 11", 2021),
    ("Layers of Fear", "Steam", "Layers of Fear 2016 video game", 2021),
    ("Super Mario Bros III", "Steam", "Super Mario Bros. 3", 2021),
    ("Horizon Zero Dawn", "PS5", None, 2021),
    ("Until Dawn", "PS5", None, 2022),
    ("The Last Cube", "Steam", None, 2022),
    ("Spider Man Miles Morales", "PS5", "Spider-Man: Miles Morales", 2022),
    ("TMNT Shredders Revenge", "PS5", "Teenage Mutant Ninja Turtles: Shredder's Revenge", 2022),
    ("Dark Anthology: Man of Medan", "PS5", "The Dark Pictures Anthology: Man of Medan", 2022),
    ("Dark Anthology: Little Hope", "PS5", "The Dark Pictures Anthology: Little Hope", 2022),
    ("Dark Anthology: House of Ashes", "PS5", "The Dark Pictures Anthology: House of Ashes", 2022),
    ("Crash 4 - It's About Time", "PS5", "Crash Bandicoot 4: It's About Time", 2022),
    ("Resident Evil 0", "PS5", "Resident Evil Zero", 2022),
    ("The Quarry", "PS5", "The Quarry video game", 2022),
    ("Stray", "PS5", "Stray 2022 video game", 2022),
    ("Little Nightmares", "PS5", None, 2022),
    ("NFS Heat", "PS5", "Need for Speed Heat", 2022),
    ("Gotham Knights", "PS5", None, 2022),
    ("The Callisto Protocol", "PS5", None, 2022),
    ("God of War Ragnarok", "PS5", "God of War Ragnarök", 2022),
    ("Inside", "PS5", "Inside video game", 2022),
    ("The Council", "PS5", "The Council video game", 2022),
    ("Nickelodeon All Stars Brawl", "PS5", "Nickelodeon All-Star Brawl", 2022),
    ("Star Wars Jedi: Fallen Order", "PS5", None, 2023),
    ("Abzu", "PS5", "Abzû", 2023),
    ("Hogwarts Legacy", "PS5", None, 2023),
    ("Resident Evil 4 Remake", "PS5", "Resident Evil 4 2023 video game", 2023),
    ("Star Wars Jedi: Survivor", "PS5", None, 2023),
    ("SF6 World Tour", "PS5", "Street Fighter 6", 2023),
    ("Diablo IV", "PS5", None, 2023),
    ("FF 16", "PS5", "Final Fantasy XVI", 2023),
    ("Mortal Kombat 1", "PS5", "Mortal Kombat 1 2023 video game", 2023),
    ("Sea of Stars", "PS5", "Sea of Stars video game", 2023),
    ("Spiderman 2", "PS5", "Marvel's Spider-Man 2", 2023),
    ("RoboCop Rogue City", "PS5", "RoboCop: Rogue City", 2023),
    ("Telling Lies", "PS5", "Telling Lies video game", 2023),
    ("Untitled Goose Game", "PS5", None, 2023),
    ("Super Mario 3D World", "Switch", None, 2023),
    ("Mario Kart 8 Deluxe", "Switch", "Mario Kart 8", 2024),
    ("Silent Hill: The Short Message", "PS5", None, 2024),
    ("Rollerdrome", "PS5", None, 2024),
    ("FF7R: Rebirth", "PS5", "Final Fantasy VII Rebirth", 2024),
    ("Alone in The Dark", "PS5", "Alone in the Dark 2024 video game", 2024),
    ("South Park: Snow Day", "PS5", "South Park: Snow Day!", 2024),
    ("Crisis Core: FF7 Reunion", "PS5", "Crisis Core: Final Fantasy VII Reunion", 2024),
    ("Streets of Rage 4", "PS5", None, 2024),
    ("Star Wars Outlaws", "PS5", None, 2024),
    ("Silent Hill 2 Remake", "PS5", "Silent Hill 2 2024 video game", 2024),
    ("Broken Sword: Reforged", "PS5", "Broken Sword: Shadow of the Templars – Reforged", 2024),
    ("Dark Anthology: The Devil in Me", "PS5", "The Dark Pictures Anthology: The Devil in Me", 2024),
    ("Mario Wonder", "Switch", "Super Mario Bros. Wonder", 2024),
    ("Donkey Kong Country: Tropical Freeze", "Switch", None, 2024),
    ("Ultimate Marvel VS Capcom 3", "PS5", "Ultimate Marvel vs. Capcom 3", 2025),
    ("Tekken 8", "PS5", None, 2025),
    ("Resident Evil 1 HD Remaster", "PS5", "Resident Evil HD Remaster", 2025),
    ("Metroid Dread", "Switch", None, 2025),
    ("Super Mario RPG Remake", "Switch", "Super Mario RPG 2023 video game", 2025),
    ("Zelda Links Awakening", "Switch", "The Legend of Zelda: Link's Awakening 2019 video game", 2025),
    ("Indiana Jones and the Great Circle", "PS5", None, 2025),
    ("Super Street Fighter 30th Anniversary", "PS5", "Street Fighter 30th Anniversary Collection", 2025),
    ("Mafia: The Old Country", "PS5", None, 2025),
    ("Ninja Gaiden Ragebound", "PS5", None, 2025),
    ("Ghost of Yotei", "PS5", "Ghost of Yōtei", 2025),
    ("Syberia: The World Before", "PS5", None, 2025),
    ("The Stanley Parable: Ultra Deluxe", "Steam", None, 2025),
    ("Contra Operation Galuga", "PS5", "Contra: Operation Galuga", 2025),
    ("The Complex", "Steam", "The Complex video game", 2025),
    ("Battlefield 6", "PS5", None, 2025),
    ("Timespinner", "Steam", None, 2025),
    ("Pools", "Steam", "Pools video game", 2025),
    ("Resident Evil Revelations 2", "PS5", None, 2026),
    ("Resident Evil Requiem", "PS5", None, 2026),
    ("007 First Light", "PS5", None, 2026),
    ("Disney Epic Mickey Rebrushed", "PS5", "Epic Mickey: Rebrushed", 2026),
]

CONSOLE_PAGES = {
    "ps5": "PlayStation 5",
    "switch": "Nintendo Switch",
    "steam": "Steam (service)",
}

PALETTES = [
    ("#1b3a6b", "#3f7fd4"),
    ("#5b2340", "#c04a76"),
    ("#1f4d3a", "#4fa376"),
    ("#6b4a1b", "#d49a3f"),
    ("#3a2a5b", "#7a5fd0"),
    ("#5b1f1f", "#c0504a"),
]


def slugify(name):
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "game"


def api(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def wiki_thumb(query, size=640):
    q = urllib.parse.quote(query)
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&format=json"
        "&generator=search&gsrlimit=1&gsrsearch=" + q +
        "&prop=pageimages&piprop=thumbnail&pilicense=any&pithumbsize=" + str(size)
    )
    data = api(url)
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        thumb = page.get("thumbnail", {}).get("source")
        if thumb:
            return thumb
    return None


def download(url, path):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    if len(data) < 500:
        raise ValueError("image too small")
    with open(path, "wb") as f:
        f.write(data)


def placeholder(name, path, idx):
    dark, light = PALETTES[idx % len(PALETTES)]
    words = name.split()
    lines, line = [], ""
    for w in words:
        if len(line + " " + w) > 14 and line:
            lines.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    lines.append(line)
    lines = lines[:5]
    y = 240 - (len(lines) - 1) * 22
    texts = ""
    for i, l in enumerate(lines):
        texts += (
            '<text x="150" y="' + str(y + i * 44) +
            '" text-anchor="middle" font-family="Georgia,serif" font-size="30" '
            'font-weight="bold" fill="#f5efe2">' + l.replace("&", "&amp;") + "</text>"
        )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400">'
        '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="' + light + '"/><stop offset="1" stop-color="' + dark + '"/>'
        "</linearGradient></defs>"
        '<rect width="300" height="400" fill="url(#g)"/>'
        '<rect x="14" y="14" width="272" height="372" fill="none" stroke="#f5efe2" stroke-width="2" opacity="0.5"/>'
        + texts +
        '<text x="150" y="368" text-anchor="middle" font-family="Georgia,serif" font-size="14" fill="#f5efe2" opacity="0.7">GAME STAND</text>'
        "</svg>"
    )
    with open(path, "w") as f:
        f.write(svg)


def ext_for(url):
    path = urllib.parse.urlparse(url).path.lower()
    for e in (".png", ".jpg", ".jpeg", ".webp"):
        if path.endswith(e):
            return e
    return ".jpg"


def fetch_covers():
    games = []
    for idx, (name, console, query, year) in enumerate(GAMES):
        slug = slugify(name)
        cover = None
        try:
            url = wiki_thumb(query or (name + " video game"))
            if url:
                fname = slug + ext_for(url)
                download(url, os.path.join(COVERS, fname))
                cover = "covers/" + fname
                print("ok " + name)
        except Exception as e:
            print("miss " + name + " " + str(e))
        if not cover:
            fname = slug + ".svg"
            placeholder(name, os.path.join(COVERS, fname), idx)
            cover = "covers/" + fname
            print("placeholder " + name)
        games.append({
            "id": slug, "name": name, "console": console,
            "year": year, "order": idx + 1, "cover": cover,
        })
        time.sleep(0.2)
    with open(os.path.join(BASE, "games.json"), "w") as f:
        json.dump(games, f, indent=2)
    print(str(len(games)) + " games written")


def fetch_consoles():
    for key, page in CONSOLE_PAGES.items():
        try:
            url = wiki_thumb(page, 800)
            if url:
                download(url, os.path.join(CONSOLES, key + ext_for(url)))
                print("ok console " + key)
        except Exception as e:
            print("miss console " + key + " " + str(e))


if __name__ == "__main__":
    os.makedirs(COVERS, exist_ok=True)
    os.makedirs(CONSOLES, exist_ok=True)
    fetch_consoles()
    fetch_covers()
