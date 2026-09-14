import re

WEIGHTS = {"critical": 7, "high": 3, "medium": 2, "low": 1}

BANDS = [
    (9, "safe", "Safe"),
    (7, "low", "Low risk"),
    (4, "review", "Needs review"),
    (0, "dangerous", "Dangerous"),
]

RULES = [
    {
        "id": "pipe-to-shell",
        "severity": "critical",
        "title": "Downloads a remote script and runs it",
        "why": "Whatever the server returns runs with your user permissions, and the server can change it after you reviewed the skill.",
        "pattern": r"\b(curl|wget)\b[^\n|]*\|\s*(sudo\s+)?(ba|z|da|k)?sh\b|\biex\b[^\n]*\b(iwr|Invoke-WebRequest)\b",
    },
    {
        "id": "obfuscated-exec",
        "severity": "critical",
        "title": "Decodes hidden content and executes it",
        "why": "Encoding the payload hides what runs from anyone reading the skill, which is the point of doing it.",
        "pattern": r"base64\s+(-d|-D|--decode)\b[^\n]*\|\s*(ba|z)?sh\b|\beval\s*\(\s*(atob|Buffer\.from)\b|\bexec\s*\(\s*(base64\.)?b64decode|\beval\s+\"?\$\(\s*echo\s+[A-Za-z0-9+/=]{16,}",
    },
    {
        "id": "prompt-injection",
        "severity": "critical",
        "title": "Tells the agent to hide actions or drop its instructions",
        "why": "A skill is loaded into the agent context as instructions. Text that asks the agent to keep things from the user turns the agent against the user.",
        "pattern": r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+instructions|\b(do\s+not|don't|never)\s+(tell|inform|mention|show|reveal)\b[^\n]{0,40}\buser\b|without\s+(telling|asking|informing|notifying)\s+the\s+user|hide\s+(this|it)\s+from\s+the\s+user",
        "flags": re.IGNORECASE,
    },
    {
        "id": "permission-bypass",
        "severity": "critical",
        "title": "Turns off agent permission prompts or the sandbox",
        "why": "Every tool call then runs without a human approving it, so one bad step deletes or sends data with nobody in the loop.",
        "pattern": r"dangerously-skip-permissions|dangerously-bypass-approvals-and-sandbox|\bbypassPermissions\b|--yolo\b|approval_policy\s*=\s*[\"']?never|sandbox_mode\s*=\s*[\"']?danger-full-access",
    },
    {
        "id": "destructive-delete",
        "severity": "critical",
        "title": "Recursively deletes the home or root directory",
        "why": "One wrong variable or one run in the wrong place wipes the machine and there is no undo.",
        "pattern": r"\brm\s+(-[a-zA-Z]+\s+)*-[a-zA-Z]*[rR][a-zA-Z]*\s+(-[a-zA-Z]+\s+)*[\"']?(/|~|\$HOME|\$\{HOME\})[\"']?/?\*?(\s|$|;|&)",
    },
    {
        "id": "credential-access",
        "severity": "high",
        "title": "Reads credential files or tokens",
        "why": "SSH keys, cloud credentials and registry tokens give access far beyond the task a skill claims to do.",
        "pattern": r"\.ssh/(id_|authorized_keys)|\.aws/credentials|\.netrc\b|\.docker/config\.json|\.kube/config|\.gnupg/|security\s+find-(generic|internet)-password|\.git-credentials|\bgh\s+auth\s+token\b|\.config/gh/hosts\.yml|\.npmrc\b|\.pypirc\b",
    },
    {
        "id": "data-exfiltration",
        "severity": "high",
        "title": "Uploads local files or streams data to a remote host",
        "why": "Posting a local file or opening a raw socket is how data leaves the machine without the user noticing.",
        "pattern": r"\bcurl\b[^\n]*\s(-d|--data(-binary|-raw)?|-F|--form|-T|--upload-file)\s+[\"']?@|/dev/tcp/|\bnc\s+[^\n]*\s\d{2,5}\s*<|\bscp\s+[^\n]*\s[\w.-]+@[\w.-]+:",
    },
    {
        "id": "raw-ip-endpoint",
        "severity": "high",
        "title": "Talks to a raw IP address",
        "why": "A public service has a domain name. A hardcoded public IP hides who receives the traffic.",
        "pattern": r"\bhttps?://(?!127\.|0\.0\.0\.0|10\.|192\.168\.|localhost)(\d{1,3}\.){3}\d{1,3}\b",
    },
    {
        "id": "persistence",
        "severity": "high",
        "title": "Installs something that survives the session",
        "why": "Cron jobs, launch agents and shell profile edits keep running long after the skill finished, every time you open a terminal.",
        "pattern": r"\bcrontab\b(?!\s+-l)|\blaunchctl\s+(load|bootstrap|enable)\b|Launch(Agents|Daemons)/|\bsystemctl\s+(--user\s+)?enable\b|>>?\s*[\"']?(~|\$HOME|\$\{HOME\})/\.(zshrc|bashrc|bash_profile|profile|zprofile)",
    },
    {
        "id": "agent-config-tamper",
        "severity": "high",
        "title": "Rewrites agent settings, hooks or global instructions",
        "why": "Changing settings, hooks or global instruction files changes how every future session behaves, not just this skill.",
        "pattern": r"(>>?|\btee\b[^\n]*|\bsed\s+-i\b[^\n]*)\s*[\"']?(~|\$HOME|\$\{HOME\})/\.(claude|codex)/(settings|config\.toml|hooks|CLAUDE\.md|AGENTS\.md)",
    },
    {
        "id": "privilege-escalation",
        "severity": "high",
        "title": "Asks for root or opens up file permissions",
        "why": "A skill does not need root to help write code. sudo and world-writable files widen what a mistake can break.",
        "pattern": r"\bsudo\s+\S|\bchmod\s+(-R\s+)?(777|[ug]?\+s)\b|\bchown\s+(-R\s+)?root\b",
    },
    {
        "id": "hidden-unicode",
        "severity": "high",
        "title": "Contains invisible or direction-flipping characters",
        "why": "Zero-width and bidi characters make the text the agent reads differ from the text a human sees in review.",
        "pattern": "[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]",
    },
    {
        "id": "env-dump",
        "severity": "medium",
        "title": "Dumps every environment variable",
        "why": "The environment usually holds API keys and tokens. Printing all of it puts them into logs and the agent context.",
        "pattern": r"\bprintenv\b|^\s*env\s*($|\|)|\bdict\(\s*os\.environ\s*\)|JSON\.stringify\(\s*process\.env\s*\)",
    },
    {
        "id": "destructive-git",
        "severity": "medium",
        "title": "Rewrites or discards git history",
        "why": "Force pushes and hard resets destroy commits other people or you still need.",
        "pattern": r"\bgit\s+push\b[^\n]*(--force\b|\s-f\b|--force-with-lease)|\bgit\s+reset\s+--hard\b|\bgit\s+clean\s+-[a-zA-Z]*f",
    },
    {
        "id": "third-party-install",
        "severity": "medium",
        "title": "Installs third-party code at run time",
        "why": "Packages fetched on the fly are not pinned or reviewed, so what runs today can differ from what runs tomorrow.",
        "pattern": r"\bnpx\s+(-y|--yes)\b|\bnpm\s+(i|install)\s+-g\b|\bpip3?\s+install\s+[^\n]*(git\+|https?://)|\bgo\s+install\s+\S+@latest\b|\bcargo\s+install\b",
    },
    {
        "id": "encoded-blob",
        "severity": "medium",
        "title": "Carries a long encoded blob",
        "why": "A long base64 string cannot be reviewed by reading it. It can be an icon or a payload.",
        "pattern": r"[A-Za-z0-9+/]{160,}={0,2}",
    },
    {
        "id": "wildcard-tools",
        "severity": "medium",
        "title": "Grants every tool without restriction",
        "why": "A wildcard in allowed-tools lets the skill run any tool without the permission prompt narrowing it.",
        "pattern": r"^\s*allowed-tools\s*:.*(^|[\s\[,\"'])\*([\s\],\"']|$)",
        "only": "SKILL.md",
    },
    {
        "id": "dynamic-exec",
        "severity": "low",
        "title": "Builds and runs commands dynamically",
        "why": "eval, exec and shell=True run strings assembled at run time, which is where injected input turns into commands.",
        "pattern": r"\beval\s+[\"']?\$|\beval\s*\(|\bexec\s*\(|shell\s*=\s*True|\bos\.system\s*\(|\bchild_process\b",
    },
    {
        "id": "network-access",
        "severity": "low",
        "title": "Makes network requests",
        "why": "Not a problem on its own. Worth knowing which skills reach the network when one of them turns bad.",
        "pattern": r"\b(curl|wget)\b|\brequests\.(get|post|put)\b|\burllib\.request\b|\bfetch\s*\(|\bhttp\.client\b|\baxios\.",
    },
]

FILE_RULES = [
    {
        "id": "executable-binary",
        "severity": "high",
        "title": "Ships a compiled executable",
        "why": "A compiled binary cannot be reviewed as text. Nothing in the skill proves what it does.",
    },
    {
        "id": "symlink-escape",
        "severity": "high",
        "title": "Links to a file outside the skill",
        "why": "A symlink out of the skill folder can point the agent at secrets or at files that change under it.",
    },
    {
        "id": "oversized-file",
        "severity": "medium",
        "title": "File too large to inspect",
        "why": "Files over 2 MB are hashed but not scanned, so a rule could miss something inside.",
    },
    {
        "id": "binary-file",
        "severity": "low",
        "title": "Contains a binary file",
        "why": "Images and archives are common in skills, but their content is not scanned by the text rules.",
    },
]

COMPILED = [(r, re.compile(r["pattern"], r.get("flags", 0))) for r in RULES]

ALL_RULES = {r["id"]: r for r in RULES + FILE_RULES}


def band(score):
    for floor, key, label in BANDS:
        if score >= floor:
            return key, label
    return BANDS[-1][1], BANDS[-1][2]


def score(findings):
    seen = {}
    for f in findings:
        key = f["rule"]
        if WEIGHTS[f["severity"]] > seen.get(key, 0):
            seen[key] = WEIGHTS[f["severity"]]
    return max(0, 10 - sum(seen.values()))


def public_rules():
    return [
        {"id": r["id"], "severity": r["severity"], "weight": WEIGHTS[r["severity"]], "title": r["title"], "why": r["why"]}
        for r in RULES + FILE_RULES
    ]
