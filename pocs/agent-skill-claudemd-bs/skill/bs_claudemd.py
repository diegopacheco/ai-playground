import html as html_mod
import json
import os
import re
import sys

STOPWORDS = {
    "a", "an", "the", "to", "of", "and", "or", "is", "are", "be", "on", "in",
    "for", "with", "as", "it", "this", "that", "you", "your", "i", "we", "will",
    "if", "when", "than", "then", "else", "but", "so", "at", "by", "from", "up",
    "out", "off", "all", "any", "each", "every", "what", "which", "how", "why",
    "can", "may", "should", "would", "could", "into", "over", "under", "more",
    "most", "less", "least", "very", "just", "not", "no", "yes", "do", "does",
    "did", "done", "make", "makes", "made", "sure", "use", "using", "used",
    "always", "never", "dont", "avoid", "prefer", "ensure", "keep", "want",
    "need", "must", "them", "they", "their", "there", "here", "one", "two",
    "set", "get", "way", "thing", "things", "etc", "via", "per", "about",
    "possible", "latest", "really", "actually", "basically", "simply",
}

VAGUE_PHRASES = [
    "as simple as possible", "simple as possible", "make sense", "makes sense",
    "well written", "well-written", "as possible", "if possible", "properly",
    "appropriate", "appropriately", "reasonable", "good enough", "make it cool",
    "make it nice", "make it visual", "make it useful", "and more", "or more",
    "keep that in mind", "as needed", "as you see fit", "best practice",
    "high quality", "clean code", "where possible", "use judgment", "and so on",
    "kind of", "sort of", "etc",
]

LOWSIGNAL_PHRASES = [
    "well written", "make sense", "makes sense", "simple as possible",
    "minimum code", "nothing speculative", "no features beyond", "senior engineer",
    "caution over speed", "think before", "read before you write", "fail loud",
    "surgical", "touch only what you must", "don't refactor", "dont refactor",
    "match existing style", "define success", "code that makes sense",
    "the right thing", "be careful", "high quality", "best practice",
    "do not do things i did not ask", "only what was asked",
]

GENERIC_OBJECTS = {
    "line", "code", "file", "version", "function", "stuff", "item", "list",
    "name", "value", "build", "step", "part", "case", "point", "stack",
}


def stem(tok):
    if tok.endswith("'s"):
        tok = tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        tok = tok[:-1]
    return tok


def tokens(text):
    raw = re.sub(r"[^a-z0-9']+", " ", text.lower()).split()
    out = []
    for t in raw:
        t = t.strip("'")
        if not t:
            continue
        out.append(t)
    return out


def significant(text):
    out = set()
    for t in tokens(text):
        s = stem(t)
        if len(s) >= 4 and s not in STOPWORDS and t not in STOPWORDS and not s.isdigit():
            out.add(s)
    return out


def clean_rule_text(raw):
    t = raw.strip()
    t = re.sub(r"^[-*+]\s+", "", t)
    t = re.sub(r"^\d+\.\s+", "", t)
    t = re.sub(r"^Rule\s+\d+\s*[—\-:]\s*", "", t, flags=re.I)
    t = t.replace("**", "").replace("`", "")
    return t.strip()


def is_heading(raw):
    return bool(re.match(r"^\s{0,3}#{1,6}\s", raw))


def heading_text(raw):
    return re.sub(r"^\s{0,3}#{1,6}\s+", "", raw).strip()


def is_rule_line(raw):
    s = raw.strip()
    if not s or is_heading(raw):
        return False
    if re.match(r"^[-*+]\s+", s) or re.match(r"^\d+\.\s+", s):
        return True
    cues = ["never", "always", "must", "do not", "don't", "dont", "avoid",
            "prefer", "ensure", "make sure", "should", "use ", "touch ",
            "define ", "state ", "stop ", "push back", "bias:", "test:"]
    low = s.lower()
    return any(c in low for c in cues)


NEG_FLIP = {"never", "don't", "dont", "avoid", "without", "not", "no", "nor"}
POS_FLIP = {"always", "must", "prefer", "should", "ensure", "shall"}


def polarity(text):
    cur = "neutral"
    for t in tokens(text):
        if t in NEG_FLIP:
            return "neg" if cur == "neutral" else cur
        if t in POS_FLIP:
            return "pos" if cur == "neutral" else cur
    return cur


def assertions(text):
    out = []
    cur = None
    for t in tokens(text):
        if t in NEG_FLIP:
            cur = "neg"
            continue
        if t in POS_FLIP:
            cur = "pos"
            continue
        if cur is None:
            continue
        s = stem(t)
        if len(s) >= 4 and s not in STOPWORDS and not s.isdigit():
            out.append((s, cur))
    return out


def parse(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    records = []
    section = "(top)"
    in_code = False
    for i, raw in enumerate(lines, 1):
        rec = {"n": i, "raw": raw, "section": section, "issues": [],
               "kind": "text", "core": "", "sig": [], "polarity": "neutral"}
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            rec["kind"] = "code"
            records.append(rec)
            continue
        if in_code:
            rec["kind"] = "code"
            records.append(rec)
            continue
        if not stripped:
            rec["kind"] = "blank"
            records.append(rec)
            continue
        if is_heading(raw):
            rec["kind"] = "heading"
            rec["core"] = heading_text(raw)
            rec["level"] = len(re.match(r"^\s*(#{1,6})", raw).group(1))
            section = rec["core"]
            records.append(rec)
            continue
        core = clean_rule_text(raw)
        rec["core"] = core
        rec["sig"] = sorted(significant(core))
        rec["polarity"] = polarity(core)
        rec["kind"] = "rule" if is_rule_line(raw) else "text"
        records.append(rec)
    return records


def find_phrases(text, phrases):
    low = text.lower()
    return [p for p in phrases if p in low]


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def evaluate(records, label):
    rules = [r for r in records if r["kind"] in ("rule", "text")]
    num_rules = len(rules)

    df = {}
    for r in rules:
        for s in set(r["sig"]):
            df[s] = df.get(s, 0) + 1

    rare = {t for t, c in df.items() if 2 <= c <= 4}

    def jacc(a, b):
        sa, sb = set(a), set(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    idx = {id(r): k for k, r in enumerate(rules)}
    uf = UF(len(rules))
    for i in range(len(rules)):
        for j in range(i + 1, len(rules)):
            a, b = rules[i], rules[j]
            shared_rare = (set(a["sig"]) & set(b["sig"])) & rare
            sim = jacc(a["sig"], b["sig"])
            if (shared_rare and sim >= 0.2) or sim >= 0.5:
                uf.union(i, j)

    comps = {}
    for r in rules:
        root = uf.find(idx[id(r)])
        comps.setdefault(root, []).append(r)

    overlaps = []
    for root, members in comps.items():
        if len(members) < 2:
            continue
        keys = {}
        for r in members:
            for s in set(r["sig"]):
                if s in rare:
                    keys[s] = keys.get(s, 0) + 1
        shared = sorted([k for k, c in keys.items() if c >= 2],
                        key=lambda k: -keys[k])
        if not shared:
            continue
        if len(members) == 2 and jacc(members[0]["sig"], members[1]["sig"]) < 0.3:
            continue
        lns = sorted(m["n"] for m in members)
        overlaps.append({"keys": shared[:4], "lines": lns,
                         "items": [{"n": m["n"], "text": m["core"],
                                    "section": m["section"]} for m in
                                   sorted(members, key=lambda m: m["n"])]})
        for m in members:
            others = [x for x in lns if x != m["n"]]
            m["issues"].append({
                "type": "overlap", "sev": 2,
                "msg": "Overlaps rule(s) " + ", ".join("L" + str(o) for o in others)
                + " on “" + shared[0] + "” — consolidate into one rule."})
    overlaps.sort(key=lambda o: -len(o["lines"]))

    for r in rules:
        text = r["core"]
        vague = find_phrases(text, VAGUE_PHRASES)
        if vague:
            r["issues"].append({
                "type": "vague", "sev": 2,
                "msg": "Vague — “" + vague[0]
                + "” has no checkable criterion; the model can't tell when it's satisfied."})
        low = find_phrases(text, LOWSIGNAL_PHRASES)
        if low and not vague:
            r["issues"].append({
                "type": "lowsignal", "sev": 1,
                "msg": "Low signal — restates behavior the model already biases toward; spends tokens without changing output much."})
        wc = len(tokens(text))
        if wc > 28:
            r["issues"].append({
                "type": "overlong", "sev": 1,
                "msg": "Long rule (" + str(wc) + " words) — split into atomic, single-purpose rules."})
        if re.search(r"\bnever\b.*\bnever\b", text, re.I) or "never ever" in text.lower():
            r["issues"].append({
                "type": "redundant", "sev": 1,
                "msg": "Redundant emphasis — repeats the same prohibition; one clear statement is enough."})

    by_token = {}
    for r in rules:
        for s, pol in assertions(r["core"]):
            slot = by_token.setdefault(s, {"pos": set(), "neg": set()})
            slot[pol].add(r["n"])
            r.setdefault("asserts", {})[s] = pol
    contradictions = []
    seen = set()
    rule_by_n = {r["n"]: r for r in rules}
    for t, sides in by_token.items():
        pos_rules = [rule_by_n[n] for n in sorted(sides["pos"])]
        neg_rules = [rule_by_n[n] for n in sorted(sides["neg"])]
        sides = {"pos": pos_rules, "neg": neg_rules}
        if sides["pos"] and sides["neg"] and t in rare and t not in GENERIC_OBJECTS:
            for a in sides["pos"]:
                for b in sides["neg"]:
                    if a["n"] == b["n"]:
                        continue
                    if jacc(a["sig"], b["sig"]) < 0.34:
                        continue
                    key = (min(a["n"], b["n"]), max(a["n"], b["n"]))
                    if key in seen:
                        continue
                    seen.add(key)
                    contradictions.append({
                        "token": t,
                        "pos": {"n": a["n"], "text": a["core"], "section": a["section"]},
                        "neg": {"n": b["n"], "text": b["core"], "section": b["section"]}})
                    for r in (a, b):
                        r["issues"].append({
                            "type": "contradiction", "sev": 3,
                            "msg": "Contradiction on “" + t + "” — conflicts with L"
                            + str(b["n"] if r is a else a["n"]) + "; the model can't honor both."})

    for r in records:
        if r["kind"] == "heading":
            r["verdict"] = "section"
            r["comment"] = "Section header."
            continue
        if r["kind"] == "blank":
            r["verdict"] = "blank"
            r["comment"] = ""
            continue
        if r["kind"] == "code":
            r["verdict"] = "code"
            r["comment"] = "Code / fenced block."
            continue
        if r["issues"]:
            worst = max(i["sev"] for i in r["issues"])
            r["verdict"] = {1: "info", 2: "warn", 3: "bad"}[worst]
            r["comment"] = " ".join(i["msg"] for i in r["issues"])
        else:
            if r["kind"] == "text":
                r["verdict"] = "ok"
                r["comment"] = "Context / prose line."
            else:
                concrete = bool(re.search(r"\d|\.sh\b|\.md\b|[A-Z][a-z]+[A-Z]|/|\bpodman\b|\bgit\b", r["core"]))
                r["verdict"] = "keep"
                r["comment"] = ("Concrete and checkable — good rule."
                                if concrete else
                                "Actionable rule — keep, but watch it isn't restated elsewhere.")

    n_dup = sum(1 for r in rules if any(i["type"] == "overlap" for i in r["issues"]))
    n_vague = sum(1 for r in rules if any(i["type"] == "vague" for i in r["issues"]))
    n_low = sum(1 for r in rules if any(i["type"] == "lowsignal" for i in r["issues"]))
    n_long = sum(1 for r in rules if any(i["type"] == "overlong" for i in r["issues"]))
    n_redund = sum(1 for r in rules if any(i["type"] == "redundant" for i in r["issues"]))
    n_contra = len(contradictions)
    n_keep = sum(1 for r in rules if r["verdict"] == "keep")
    n_words = sum(len(tokens(r["core"])) for r in records if r["kind"] != "blank")
    est_tokens = int(round(n_words * 1.33))

    dup_extra_words = 0
    for o in overlaps:
        for it in o["items"][1:]:
            dup_extra_words += len(tokens(it["text"]))
    low_words = sum(len(tokens(r["core"])) for r in rules
                    if any(i["type"] == "lowsignal" for i in r["issues"]))
    wasted_tokens = int(round((dup_extra_words + low_words) * 1.33))

    dup_ratio = n_dup / num_rules if num_rules else 0
    vague_ratio = n_vague / num_rules if num_rules else 0
    low_ratio = n_low / num_rules if num_rules else 0
    bs = 100 * (0.45 * dup_ratio + 0.30 * vague_ratio + 0.25 * low_ratio) + 9 * n_contra + 2 * n_redund
    bs = max(0, min(100, int(round(bs))))
    grade = ("A" if bs < 15 else "B" if bs < 30 else "C" if bs < 45
             else "D" if bs < 65 else "F")

    counts = {"keep": 0, "ok": 0, "info": 0, "warn": 0, "bad": 0}
    for r in rules:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    return {
        "label": label,
        "lines": [{"n": r["n"], "raw": r["raw"], "kind": r["kind"],
                   "section": r["section"], "verdict": r.get("verdict", "ok"),
                   "comment": r.get("comment", ""), "level": r.get("level", 0),
                   "core": r.get("core", "")} for r in records],
        "overlaps": overlaps,
        "contradictions": contradictions,
        "metrics": {
            "total_lines": len(records),
            "rules": num_rules,
            "sections": sum(1 for r in records if r["kind"] == "heading"),
            "duplicates": n_dup,
            "overlap_groups": len(overlaps),
            "vague": n_vague,
            "lowsignal": n_low,
            "overlong": n_long,
            "redundant": n_redund,
            "contradictions": n_contra,
            "keep": n_keep,
            "est_tokens": est_tokens,
            "wasted_tokens": wasted_tokens,
            "bs": bs, "grade": grade,
        },
        "dist": counts,
    }


def cross_contradictions(a, b):
    def index(rep):
        m = {}
        dfm = {}
        for ln in rep["lines"]:
            if ln["kind"] not in ("rule", "text"):
                continue
            seen = {}
            for s, pol in assertions(ln["core"]):
                seen.setdefault(s, pol)
            for s, pol in seen.items():
                if s in GENERIC_OBJECTS:
                    continue
                m.setdefault(s, {"pos": [], "neg": []})[pol].append(ln)
                dfm[s] = dfm.get(s, 0) + 1
        return m, dfm

    ia, dfa = index(a)
    ib, dfb = index(b)
    out = []
    for t in set(ia) & set(ib):
        for pol_a, pol_b in (("pos", "neg"), ("neg", "pos")):
            for ra in ia[t][pol_a]:
                for rb in ib[t][pol_b]:
                    out.append({
                        "token": t, "rank": dfa.get(t, 9) + dfb.get(t, 9),
                        "a": {"n": ra["n"], "text": ra["core"],
                              "section": ra["section"], "pol": pol_a},
                        "b": {"n": rb["n"], "text": rb["core"],
                              "section": rb["section"], "pol": pol_b}})
    best = {}
    for c in out:
        key = (c["a"]["n"], c["b"]["n"])
        if key not in best or c["rank"] < best[key]["rank"]:
            best[key] = c
    uniq = sorted(best.values(), key=lambda c: (c["rank"], c["a"]["n"], c["b"]["n"]))
    for c in uniq:
        c.pop("rank", None)
    return uniq


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ · CLAUDE.md BS Detector</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Caveat:wght@600;700&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{
 --bg:#f5f6fb; --panel:#ffffff; --ink:#1d2230; --muted:#6b7280; --line:#e8eaf2;
 --shadow:0 1px 2px rgba(28,32,54,.04),0 12px 34px rgba(28,32,54,.08);
 --keep:#1f9d63; --ok:#8a93a6; --info:#3b82f6; --warn:#dd8b1e; --bad:#e0533d;
 --accent:#6d5cf0; --accent2:#ff7aa8;
}
*{box-sizing:border-box}
body{margin:0;background:
 radial-gradient(1100px 460px at 12% -8%,#efeaff 0,rgba(245,246,251,0) 60%),
 radial-gradient(900px 420px at 100% 0,#ffeef4 0,rgba(245,246,251,0) 55%),
 var(--bg);
 color:var(--ink);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
 -webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:34px 22px 70px}
.hd{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;flex-wrap:wrap;margin-bottom:22px}
.brand .kick{font-family:Caveat,cursive;font-size:26px;color:var(--accent);line-height:.9;transform:rotate(-2deg);display:inline-block}
.brand h1{margin:2px 0 0;font-size:30px;font-weight:800;letter-spacing:-.02em}
.brand .sub{color:var(--muted);font-size:13px;margin-top:6px}
.brand .sub b{color:var(--ink)}
.gradewrap{display:flex;align-items:center;gap:16px}
.gauge{position:relative;width:200px;height:118px}
.gauge .lab{position:absolute;left:0;right:0;bottom:2px;text-align:center;font-family:Caveat,cursive;font-size:18px;color:var(--muted)}
.gauge .val{position:absolute;left:0;right:0;top:46px;text-align:center;font-size:34px;font-weight:800}
.gauge .val small{font-size:13px;color:var(--muted);font-weight:600}
.gradebadge{width:76px;height:76px;border-radius:20px;display:grid;place-items:center;font-size:40px;font-weight:800;color:#fff;box-shadow:var(--shadow)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:6px 0 22px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:15px;padding:14px 15px;box-shadow:var(--shadow)}
.card .k{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;font-weight:700}
.card .v{font-size:30px;font-weight:800;margin-top:3px;line-height:1}
.card .n{font-size:11.5px;color:var(--muted);margin-top:5px}
.card.bad .v{color:var(--bad)} .card.warn .v{color:var(--warn)} .card.good .v{color:var(--keep)}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:16px}
.tab{border:1px solid var(--line);background:var(--panel);color:var(--muted);padding:9px 16px;border-radius:11px;font-size:13px;font-weight:700;cursor:pointer}
.tab.on{color:#fff;background:var(--accent);border-color:var(--accent);box-shadow:0 6px 16px rgba(109,92,240,.28)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:20px 22px;box-shadow:var(--shadow);margin-bottom:20px}
.panel h2{margin:0 0 4px;font-size:17px;font-weight:800}
.panel .desc{color:var(--muted);font-size:13px;margin:0 0 16px}
.view{display:none} .view.on{display:block}
.toolbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:14px}
.toolbar input{flex:1;min-width:180px;border:1px solid var(--line);border-radius:10px;padding:9px 12px;font-size:13px;font-family:inherit;background:#fbfbfe}
.chip{border:1px solid var(--line);background:#fbfbfe;color:var(--muted);padding:6px 11px;border-radius:9px;font-size:12px;font-weight:700;cursor:pointer}
.chip.on{color:#fff;border-color:transparent}
.chip[data-f=all].on{background:var(--ink)} .chip[data-f=bad].on{background:var(--bad)}
.chip[data-f=warn].on{background:var(--warn)} .chip[data-f=info].on{background:var(--info)}
.chip[data-f=keep].on{background:var(--keep)}
.code{font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace;font-size:12.5px}
.row{display:grid;grid-template-columns:42px 1fr;border-radius:10px;margin-bottom:2px;overflow:hidden}
.row .ln{color:#aeb4c4;text-align:right;padding:7px 10px 7px 4px;user-select:none;background:#fafbff;border-right:1px solid var(--line)}
.row .body{padding:7px 12px;min-width:0}
.row .src{white-space:pre-wrap;word-break:break-word}
.row .cmt{font-family:Inter,sans-serif;font-size:11.5px;margin-top:4px;display:flex;gap:7px;align-items:flex-start}
.row .cmt .dot{flex:0 0 auto;width:8px;height:8px;border-radius:50%;margin-top:3px}
.row.heading{margin-top:14px}
.row.heading .src{font-weight:800;color:var(--accent)}
.row.blank .body{height:14px}
.row.v-keep{background:rgba(31,157,99,.05)} .row.v-keep .body{border-left:3px solid var(--keep)}
.row.v-ok .body{border-left:3px solid var(--ok)}
.row.v-info{background:rgba(59,130,246,.05)} .row.v-info .body{border-left:3px solid var(--info)}
.row.v-warn{background:rgba(221,139,30,.07)} .row.v-warn .body{border-left:3px solid var(--warn)}
.row.v-bad{background:rgba(224,83,61,.07)} .row.v-bad .body{border-left:3px solid var(--bad)}
.tag{display:inline-block;font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.05em;padding:2px 7px;border-radius:999px;color:#fff;margin-left:8px;vertical-align:middle}
.t-keep{background:var(--keep)} .t-ok{background:var(--ok)} .t-info{background:var(--info)}
.t-warn{background:var(--warn)} .t-bad{background:var(--bad)}
.grp{border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin-bottom:12px;background:#fcfcff}
.grp .gh{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.grp .gh .key{font-family:Caveat,cursive;font-size:22px;color:var(--accent);font-weight:700}
.grp .gi{display:grid;grid-template-columns:46px 1fr;gap:0 10px;padding:7px 0;border-top:1px dashed var(--line);font-size:13px}
.grp .gi .gl{color:var(--muted);font-weight:700;font-size:12px}
.grp .gi .gs{color:var(--muted);font-size:11px;margin-top:2px}
.cx{border:1px solid #f3c7bf;background:#fff7f5;border-radius:14px;padding:14px 16px;margin-bottom:12px}
.cx .ch{font-weight:800;font-size:14px;margin-bottom:10px;display:flex;gap:8px;align-items:center}
.cx .ch .tok{font-family:Caveat,cursive;font-size:22px;color:var(--bad)}
.cx .side{display:grid;grid-template-columns:70px 1fr;gap:10px;padding:7px 0;font-size:13px;align-items:start}
.cx .side .pl{font-size:10px;font-weight:800;text-transform:uppercase;color:#fff;padding:3px 8px;border-radius:999px;text-align:center;height:fit-content}
.cx .side.pos .pl{background:var(--keep)} .cx .side.neg .pl{background:var(--bad)}
.cx .vs{text-align:center;color:var(--bad);font-family:Caveat,cursive;font-size:20px;margin:2px 0}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-top:6px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{width:11px;height:11px;border-radius:3px;display:inline-block}
.empty{color:var(--muted);font-size:13px;padding:18px;text-align:center;border:1px dashed var(--line);border-radius:12px}
.foot{color:var(--muted);font-size:12px;text-align:center;margin-top:26px;line-height:1.6}
.donut{display:flex;align-items:center;gap:20px;flex-wrap:wrap}
.donut .lg{font-size:13px}
.donut .lg div{display:flex;align-items:center;gap:8px;margin:5px 0}
.donut .lg i{width:11px;height:11px;border-radius:3px}
.donut .lg b{font-variant-numeric:tabular-nums}
.summary{font-size:14px;line-height:1.65}
.summary b{color:var(--accent)}
</style>
</head>
<body>
<div class="wrap">
 <div class="hd">
  <div class="brand">
   <span class="kick">is your CLAUDE.md full of&hellip;</span>
   <h1>CLAUDE.md BS Detector</h1>
   <div class="sub" id="subline"></div>
  </div>
  <div class="gradewrap">
   <div class="gauge" id="gauge"></div>
   <div class="gradebadge" id="gradebadge"></div>
  </div>
 </div>
 <div class="cards" id="cards"></div>
 <div class="tabs" id="tabs"></div>
 <div id="views"></div>
 <div class="foot">
  Heuristic, opinionated, and proudly imperfect &mdash; it flags <i>candidates</i>, the judgment stays yours.<br>
  A rule is &ldquo;BS&rdquo; here when it's vague, duplicated, contradictory, or just restates what the model already does.
 </div>
</div>
<script>
const DATA = /*__DATA__*/null;
const VC={keep:'#1f9d63',ok:'#8a93a6',info:'#3b82f6',warn:'#dd8b1e',bad:'#e0533d'};
const VLAB={keep:'keep',ok:'prose',info:'minor',warn:'review',bad:'conflict'};
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

function gauge(bs,grade){
 const a=Math.PI*(1-bs/100), cx=100,cy=104,r=84;
 const x=cx+r*Math.cos(a), y=cy-r*Math.sin(a);
 const arc=(f,t,col,w)=>{const a0=Math.PI*(1-f/100),a1=Math.PI*(1-t/100);
  return `<path d="M ${cx+r*Math.cos(a0)} ${cy-r*Math.sin(a0)} A ${r} ${r} 0 0 1 ${cx+r*Math.cos(a1)} ${cy-r*Math.sin(a1)}" stroke="${col}" stroke-width="${w}" fill="none" stroke-linecap="round"/>`;};
 const gc=bs<15?VC.keep:bs<30?'#7cc14e':bs<45?VC.warn:bs<65?'#e8743a':VC.bad;
 document.getElementById('gauge').innerHTML=
  `<svg width="200" height="118" viewBox="0 0 200 118">
    <defs><filter id="wob"><feTurbulence type="fractalNoise" baseFrequency="0.018" numOctaves="2" seed="7" result="n"/><feDisplacementMap in="SourceGraphic" in2="n" scale="2.2"/></filter></defs>
    <g filter="url(#wob)">
     ${arc(0,100,'#eceef6',13)}
     ${arc(0,bs,gc,13)}
     <circle cx="${x}" cy="${y}" r="6.5" fill="#fff" stroke="${gc}" stroke-width="3"/>
    </g>
   </svg>
   <div class="val">${bs}<small>/100 BS</small></div>
   <div class="lab">the BS-o-meter</div>`;
 const gb=document.getElementById('gradebadge');
 gb.textContent=grade; gb.style.background=gc;
}

function donut(dist){
 const order=['bad','warn','info','keep','ok'];
 const tot=order.reduce((s,k)=>s+(dist[k]||0),0)||1;
 let off=0,segs='';
 const R=54,C=2*Math.PI*R;
 order.forEach(k=>{const v=dist[k]||0;const len=C*v/tot;
  segs+=`<circle cx="70" cy="70" r="${R}" fill="none" stroke="${VC[k]}" stroke-width="22" stroke-dasharray="${len} ${C-len}" stroke-dashoffset="${-off}" transform="rotate(-90 70 70)"/>`;
  off+=len;});
 const lg=order.map(k=>`<div><i style="background:${VC[k]}"></i>${VLAB[k]} <b>${dist[k]||0}</b></div>`).join('');
 return `<div class="donut"><svg width="140" height="140" viewBox="0 0 140 140">${segs}<circle cx="70" cy="70" r="43" fill="#fff"/><text x="70" y="66" text-anchor="middle" font-size="26" font-weight="800" fill="#1d2230">${tot}</text><text x="70" y="84" text-anchor="middle" font-size="10" fill="#6b7280">rules</text></svg><div class="lg">${lg}</div></div>`;
}

function cards(m){
 const def=[
  ['Rules',m.rules,'normative lines',''],
  ['Overlaps',m.duplicates,m.overlap_groups+' groups',m.duplicates?'warn':'good'],
  ['Vague',m.vague,'no clear criterion',m.vague?'warn':'good'],
  ['Low signal',m.lowsignal,'model already does it',m.lowsignal?'warn':'good'],
  ['Contradictions',m.contradictions,'cannot honor both',m.contradictions?'bad':'good'],
  ['Est. tokens',m.est_tokens,'~loaded every turn',''],
  ['Wasted tokens',m.wasted_tokens,'dupes + low signal',m.wasted_tokens>40?'warn':'good'],
  ['Keepers',m.keep,'concrete & checkable','good'],
 ];
 document.getElementById('cards').innerHTML=def.map(d=>
  `<div class="card ${d[3]}"><div class="k">${d[0]}</div><div class="v">${d[1]}</div><div class="n">${d[2]}</div></div>`).join('');
}

function summaryText(d){
 const m=d.metrics;
 let s=`Scanned <b>${m.total_lines}</b> lines &mdash; <b>${m.rules}</b> rules across <b>${m.sections}</b> sections. `;
 const bits=[];
 if(m.contradictions) bits.push(`<b>${m.contradictions}</b> contradiction${m.contradictions>1?'s':''} the model literally cannot satisfy`);
 if(m.overlap_groups) bits.push(`<b>${m.overlap_groups}</b> overlap group${m.overlap_groups>1?'s':''} saying the same thing more than once`);
 if(m.vague) bits.push(`<b>${m.vague}</b> vague rule${m.vague>1?'s':''} with no checkable criterion`);
 if(m.lowsignal) bits.push(`<b>${m.lowsignal}</b> low-signal line${m.lowsignal>1?'s':''} that just restate the model's defaults`);
 if(bits.length){ s+='Found '+bits.join(', ')+'. '; }
 else { s+='Clean &mdash; no overlaps, contradictions, or empty platitudes worth flagging. '; }
 s+=`Roughly <b>${m.wasted_tokens}</b> of <b>${m.est_tokens}</b> tokens look spendable. Grade <b>${m.grade}</b>.`;
 return s;
}

function lineRow(l){
 if(l.kind==='blank') return `<div class="row blank" data-v="ok" data-t=""><div class="ln"></div><div class="body"></div></div>`;
 const v=l.verdict, cls=(l.kind==='heading')?'heading':'v-'+v;
 const tag=(l.kind==='heading'||l.kind==='code')?'':`<span class="tag t-${v}">${VLAB[v]}</span>`;
 const cmt=l.comment?`<div class="cmt"><span class="dot" style="background:${VC[v]}"></span><span>${esc(l.comment)}</span></div>`:'';
 const txt=l.raw===''?' ':esc(l.raw);
 return `<div class="row ${cls}" data-v="${v}" data-t="${esc((l.raw+' '+l.comment).toLowerCase())}">
   <div class="ln">${l.n}</div>
   <div class="body"><div class="code src">${txt}${tag}</div>${cmt}</div></div>`;
}

function overlapsView(d){
 if(!d.overlaps.length) return `<div class="empty">No overlapping rules &mdash; nothing is stated twice. </div>`;
 return d.overlaps.map(o=>`<div class="grp">
   <div class="gh"><span class="key">${esc(o.keys.join(' / '))}</span><span class="tag t-warn">${o.items.length} rules</span></div>
   ${o.items.map(it=>`<div class="gi"><div class="gl">L${it.n}</div><div><div>${esc(it.text)}</div><div class="gs">${esc(it.section)}</div></div></div>`).join('')}
  </div>`).join('');
}

function contraView(list){
 if(!list.length) return `<div class="empty">No contradictions detected. The rules don't fight each other. </div>`;
 return list.map(c=>{
  const a=c.pos||c.a, b=c.neg||c.b;
  const ap=(c.pos?'pos':a.pol), bp=(c.neg?'neg':b.pol);
  return `<div class="cx"><div class="ch"><span class="tok">${esc(c.token)}</span><span>conflicting rules</span></div>
   <div class="side ${ap}"><span class="pl">${ap==='pos'?'do':"don't"}</span><div>L${a.n} &mdash; ${esc(a.text)}<div class="gs" style="color:#9aa0ad;font-size:11px">${esc(a.section)}</div></div></div>
   <div class="vs">&times;</div>
   <div class="side ${bp}"><span class="pl">${bp==='pos'?'do':"don't"}</span><div>L${b.n} &mdash; ${esc(b.text)}<div class="gs" style="color:#9aa0ad;font-size:11px">${esc(b.section)}</div></div></div>
  </div>`;}).join('');
}

function build(){
 const d=DATA.primary||DATA;
 document.getElementById('subline').innerHTML=DATA.mode==='diff'
  ? `Diffing <b>${esc(DATA.primary.label)}</b> against <b>${esc(DATA.secondary.label)}</b>`
  : `Auditing <b>${esc(d.label)}</b>`;
 gauge(d.metrics.bs,d.metrics.grade);
 cards(d.metrics);

 const tabs=[['overview','Overview'],['lines','Line&#8209;by&#8209;line'],['overlaps','Overlaps'],['contra','Contradictions']];
 if(DATA.mode==='diff') tabs.push(['diff','Cross&#8209;file conflicts']);
 document.getElementById('tabs').innerHTML=tabs.map((t,i)=>`<div class="tab ${i===0?'on':''}" data-tab="${t[0]}">${t[1]}</div>`).join('');

 const V=document.getElementById('views');
 V.innerHTML=`
  <div class="view on" data-view="overview">
   <div class="panel"><h2>The verdict</h2><p class="summary" id="sumtxt"></p>
    <div class="legend"><span><i style="background:${VC.bad}"></i>conflict</span><span><i style="background:${VC.warn}"></i>review</span><span><i style="background:${VC.info}"></i>minor</span><span><i style="background:${VC.keep}"></i>keep</span><span><i style="background:${VC.ok}"></i>prose</span></div>
   </div>
   <div class="panel"><h2>Rule health</h2><p class="desc">How every normative line scored.</p>${donut(d.dist)}</div>
  </div>
  <div class="view" data-view="lines">
   <div class="panel"><h2>Line&#8209;by&#8209;line</h2><p class="desc">Every line of the file, annotated. Filter or search to focus.</p>
    <div class="toolbar">
     <input id="q" placeholder="search lines and comments&hellip;">
     <span class="chip on" data-f="all">all</span>
     <span class="chip" data-f="bad">conflict</span>
     <span class="chip" data-f="warn">review</span>
     <span class="chip" data-f="info">minor</span>
     <span class="chip" data-f="keep">keep</span>
    </div>
    <div class="code" id="listing">${d.lines.map(lineRow).join('')}</div>
   </div>
  </div>
  <div class="view" data-view="overlaps"><div class="panel"><h2>Overlapping rules</h2><p class="desc">Same instruction, said more than once &mdash; pick one home for it.</p>${overlapsView(d)}</div></div>
  <div class="view" data-view="contra"><div class="panel"><h2>Contradictions</h2><p class="desc">Rules whose plain reading conflicts.</p>${contraView(d.contradictions)}</div></div>
  ${DATA.mode==='diff'?`<div class="view" data-view="diff"><div class="panel"><h2>Cross&#8209;file conflicts</h2><p class="desc">Where <b>${esc(DATA.primary.label)}</b> and <b>${esc(DATA.secondary.label)}</b> tell the model opposite things.</p>${contraView(DATA.cross)}</div></div>`:''}
 `;
 document.getElementById('sumtxt').innerHTML=summaryText(d);

 document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('on'));
  t.classList.add('on');
  document.querySelectorAll('.view').forEach(v=>v.classList.toggle('on',v.dataset.view===t.dataset.tab));
 });

 let filt='all';
 const apply=()=>{const q=(document.getElementById('q').value||'').toLowerCase();
  document.querySelectorAll('#listing .row').forEach(r=>{
   const v=r.dataset.v, t=r.dataset.t;
   const okF=filt==='all'||v===filt;
   const okQ=!q||t.indexOf(q)>=0;
   r.style.display=(okF&&okQ)?'':'none';
  });};
 document.getElementById('q').oninput=apply;
 document.querySelectorAll('.chip').forEach(c=>c.onclick=()=>{
  document.querySelectorAll('.chip').forEach(x=>x.classList.remove('on'));
  c.classList.add('on'); filt=c.dataset.f; apply();});
}
build();
</script>
</body>
</html>
"""


def render(report, title):
    out = TEMPLATE.replace("__TITLE__", html_mod.escape(title))
    out = out.replace("/*__DATA__*/null", json.dumps(report, ensure_ascii=False))
    return out


def write_site(report, out_dir, title):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "index.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render(report, title))
    with open(os.path.join(out_dir, "data.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return path


def cmd_analyze(args):
    if not args:
        print("usage: bs_claudemd.py analyze <CLAUDE.md> <out-dir> [title]")
        return 2
    src = args[0]
    out_dir = args[1] if len(args) > 1 else "bs-report"
    title = args[2] if len(args) > 2 else os.path.basename(os.path.dirname(os.path.abspath(src))) or "CLAUDE.md"
    if not os.path.isfile(src):
        print("file not found: " + src)
        return 1
    rep = evaluate(parse(src), os.path.basename(src) + " (" + src + ")")
    report = {"mode": "analyze", "primary": rep, "source": src}
    path = write_site(report, out_dir, title)
    m = rep["metrics"]
    print("BS score: " + str(m["bs"]) + "/100  grade " + m["grade"])
    print("rules=" + str(m["rules"]) + " overlaps=" + str(m["overlap_groups"])
          + " vague=" + str(m["vague"]) + " lowsignal=" + str(m["lowsignal"])
          + " contradictions=" + str(m["contradictions"]))
    print("wasted~" + str(m["wasted_tokens"]) + " of ~" + str(m["est_tokens"]) + " tokens")
    print("site: " + path)
    return 0


def cmd_diff(args):
    if len(args) < 2:
        print("usage: bs_claudemd.py diff <A.md> <B.md> <out-dir> [title]")
        return 2
    a_path, b_path = args[0], args[1]
    out_dir = args[2] if len(args) > 2 else "bs-report"
    title = args[3] if len(args) > 3 else "CLAUDE.md diff"
    for p in (a_path, b_path):
        if not os.path.isfile(p):
            print("file not found: " + p)
            return 1
    a = evaluate(parse(a_path), os.path.basename(a_path))
    b = evaluate(parse(b_path), os.path.basename(b_path))
    cross = cross_contradictions(a, b)
    report = {"mode": "diff", "primary": a, "secondary": b, "cross": cross,
              "source": a_path, "source_b": b_path}
    path = write_site(report, out_dir, title)
    print("cross-file contradictions: " + str(len(cross)))
    for c in cross:
        print("  · " + c["token"] + ": A:L" + str(c["a"]["n"]) + " ("
              + c["a"]["pol"] + ") vs B:L" + str(c["b"]["n"]) + " (" + c["b"]["pol"] + ")")
    print("site: " + path)
    return 0


def main(argv):
    if not argv:
        print("usage:")
        print("  bs_claudemd.py analyze <CLAUDE.md> <out-dir> [title]")
        print("  bs_claudemd.py diff <A.md> <B.md> <out-dir> [title]")
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "analyze":
        return cmd_analyze(rest)
    if cmd == "diff":
        return cmd_diff(rest)
    if os.path.isfile(cmd):
        return cmd_analyze(argv)
    print("unknown command: " + cmd)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
