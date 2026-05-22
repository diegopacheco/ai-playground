from pathlib import Path

def detect_target(repo: Path):
    repo = Path(repo)
    if (repo / "manage.py").exists() and _has_django_dep(repo):
        return "python3-django"
    if (repo / "pyproject.toml").exists() or (repo / "requirements.txt").exists() or (repo / "setup.py").exists():
        if _has_django_dep(repo):
            return "python3-django"
        return "python3"
    if (repo / "package.json").exists():
        return "nodejs"
    if (repo / "build.sbt").exists():
        return "scala3-sbt"
    if (repo / "WORKSPACE").exists() or (repo / "WORKSPACE.bazel").exists() or (repo / "MODULE.bazel").exists():
        if _has_scala_bazel(repo):
            return "scala2-bazel"
    if _has_kotlin(repo):
        return "kotlin"
    pom = repo / "pom.xml"
    if pom.exists():
        return _java_version_pom(pom)
    for name in ["build.gradle", "build.gradle.kts"]:
        g = repo / name
        if g.exists():
            return _java_version_gradle(g)
    return None

def _has_django_dep(repo: Path) -> bool:
    for name in ["requirements.txt", "pyproject.toml", "setup.py", "Pipfile"]:
        p = repo / name
        if p.exists():
            try:
                if "django" in p.read_text().lower():
                    return True
            except Exception:
                pass
    return False

def _has_kotlin(repo: Path) -> bool:
    if any((repo / n).exists() for n in ["build.gradle.kts"]):
        try:
            text = (repo / "build.gradle.kts").read_text()
            if "kotlin" in text.lower():
                return True
        except Exception:
            pass
    for f in list(repo.rglob("*.kt"))[:5]:
        return True
    return False

def _has_scala_bazel(repo: Path) -> bool:
    for f in list(repo.rglob("BUILD"))[:50] + list(repo.rglob("BUILD.bazel"))[:50]:
        try:
            if "scala_" in f.read_text():
                return True
        except Exception:
            pass
    return False

def _java_version_pom(pom: Path) -> str:
    text = pom.read_text()
    if "<source>25</source>" in text or "<maven.compiler.source>25</maven.compiler.source>" in text or "<release>25</release>" in text:
        return "java25"
    if "<source>1.8</source>" in text or "<maven.compiler.source>8</maven.compiler.source>" in text or "<release>8</release>" in text:
        return "java8"
    return "java8"

def _java_version_gradle(g: Path) -> str:
    text = g.read_text()
    if "JavaLanguageVersion.of(25)" in text or "sourceCompatibility = '25'" in text:
        return "java25"
    return "java8"
