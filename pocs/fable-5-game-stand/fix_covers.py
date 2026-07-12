import json
import os
import urllib.parse
import urllib.request

import catalog

FIXES = {
    "the-last-cube": "The Last Cube",
    "tekken-8": "Tekken 8",
    "streets-of-rage-4": "Streets of Rage 4",
    "resident-evil-1-hd-remaster": "Resident Evil (2002 video game)",
    "crisis-core-ff7-reunion": "Crisis Core: Final Fantasy VII Reunion",
    "disney-epic-mickey-rebrushed": "Epic Mickey: Rebrushed",
    "super-mario-rpg-remake": "Super Mario RPG (2023 video game)",
}


def title_thumb(title, size=640):
    q = urllib.parse.quote(title)
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&format=json&redirects=1"
        "&titles=" + q +
        "&prop=pageimages&piprop=thumbnail&pilicense=any&pithumbsize=" + str(size)
    )
    data = catalog.api(url)
    for page in data.get("query", {}).get("pages", {}).values():
        thumb = page.get("thumbnail", {}).get("source")
        if thumb:
            return thumb
    return None


def main():
    with open(os.path.join(catalog.BASE, "games.json")) as f:
        games = json.load(f)
    by_id = {g["id"]: g for g in games}
    for slug, title in FIXES.items():
        game = by_id.get(slug)
        if not game:
            print("skip " + slug)
            continue
        url = title_thumb(title)
        if not url:
            print("no image " + slug)
            continue
        old = os.path.join(catalog.BASE, "static", game["cover"])
        fname = slug + catalog.ext_for(url)
        catalog.download(url, os.path.join(catalog.COVERS, fname))
        new_cover = "covers/" + fname
        if game["cover"] != new_cover and os.path.exists(old):
            os.remove(old)
        game["cover"] = new_cover
        print("fixed " + slug)
    with open(os.path.join(catalog.BASE, "games.json"), "w") as f:
        json.dump(games, f, indent=2)


if __name__ == "__main__":
    main()
