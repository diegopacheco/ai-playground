from pathlib import Path
import subprocess

SNAPSHOT_PARENT = "__snapshot_parent__"
SNAPSHOT_HEAD = "__snapshot_head__"

class BaseRunner:
    target = "base"
    mode = "git"
    parent_dir = ".parent"

    def dodgy_diff(self, repo, diff, rdir, max_tests):
        return []

    def intent_aware(self, repo, diff, rdir, intent, max_tests):
        return []

    def _git(self, repo, *args):
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True, stderr=subprocess.STDOUT
        )

    def _diff_files(self, repo, diff, extensions):
        if self.mode == "snapshot":
            return self._diff_files_snapshot(repo, extensions)
        return self._diff_files_git(repo, diff, extensions)

    def _diff_files_git(self, repo, diff, extensions):
        try:
            if ".." in diff:
                base, head = diff.split("..", 1)
            else:
                base, head = f"{diff}~1", diff
            out = self._git(repo, "diff", "--name-only", base, head)
        except Exception:
            try:
                out = self._git(repo, "diff", "--name-only", "HEAD~1", "HEAD")
            except Exception:
                return []
        files = [f.strip() for f in out.splitlines() if f.strip()]
        return [f for f in files if any(f.endswith(e) for e in extensions)]

    def _diff_files_snapshot(self, repo, extensions):
        pdir = Path(repo) / self.parent_dir
        if not pdir.exists():
            return []
        out = []
        for p in pdir.rglob("*"):
            if not p.is_file():
                continue
            if p.name == "INTENT":
                continue
            if not any(p.name.endswith(e) for e in extensions):
                continue
            rel = str(p.relative_to(pdir))
            head_path = Path(repo) / rel
            if not head_path.exists():
                continue
            try:
                if p.read_text() != head_path.read_text():
                    out.append(rel)
            except Exception:
                continue
        return out

    def _read_at_rev(self, repo, rev, path):
        if rev == SNAPSHOT_PARENT:
            f = Path(repo) / self.parent_dir / path
            return f.read_text() if f.exists() else None
        if rev == SNAPSHOT_HEAD:
            f = Path(repo) / path
            return f.read_text() if f.exists() else None
        try:
            return self._git(repo, "show", f"{rev}:{path}")
        except Exception:
            return None

    def _parent_head(self, diff):
        if self.mode == "snapshot":
            return SNAPSHOT_PARENT, SNAPSHOT_HEAD
        if ".." in diff:
            return diff.split("..", 1)
        return f"{diff}~1", diff
