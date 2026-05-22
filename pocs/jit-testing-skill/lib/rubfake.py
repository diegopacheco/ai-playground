import re

FP_PATTERNS = [
    ("broken_test_runner", r"(surefire|test runner|junit runner).*(error|failed)"),
    ("reflection", r"java\.lang\.reflect\.|getDeclared(Method|Field|Constructor)"),
    ("type_mismatch", r"(ClassCastException|incompatible types|TypeError: )"),
    ("mock_broken", r"(MockitoException|UnfinishedStubbing|mock.*not.*configured)"),
    ("bad_mock_smell", r"(when\(.*\)\.thenReturn|mock\(.*\)\.expects)"),
    ("should_be_private_smell", r"(assertEquals\(.*private|reflectively access)"),
    ("method_must_be_protected_smell", r"protected.*invocation"),
    ("not_implemented_exception", r"(NotImplementedError|UnsupportedOperationException)"),
    ("key_value_pair_change", r"expected order of keys"),
    ("undefined_variable", r"(NameError|cannot find symbol|undefined name)"),
    ("expecting_particular_calls_to_functions", r"(verify\(.*times\(|called once)"),
    ("web_server_down", r"(ConnectionRefused|Connection refused|host.*unreachable)"),
    ("flakiness", r"(flaky|race condition|nondeterministic)"),
]

TP_PATTERNS = [
    ("changed_bool", r"expected.*<(true|false)>.*but.*<(true|false)>"),
    ("null_value", r"(NullPointerException|None.*expected|null.*not expected)"),
    ("empty_container", r"(IndexOutOfBounds|empty.*container|index.*out of range)"),
    ("unexpected_key_change", r"(KeyError|NoSuchElement|missing key)"),
    ("create_failure", r"(cannot instantiate|constructor.*failed|InstantiationException)"),
    ("rbac_change", r"(access denied|AuthorizationException|permission denied)"),
]

def score_test(catch: dict) -> dict:
    haystack = "\n".join([
        catch.get("trace") or "",
        catch.get("test_code") or "",
        catch.get("sense_check") or "",
    ])
    fp = [name for name, pat in FP_PATTERNS if re.search(pat, haystack, re.IGNORECASE)]
    tp = [name for name, pat in TP_PATTERNS if re.search(pat, haystack, re.IGNORECASE)]

    intent_title = (catch.get("intent_title") or "").lower()
    behavior_input = (catch.get("behavior_input") or "").lower()

    if intent_title:
        keywords = [w for w in re.split(r"\W+", intent_title) if len(w) > 3]
        touches_intent = any(k in behavior_input for k in keywords)
        if not touches_intent and "changed_bool" in (catch.get("kind") or ""):
            tp.append("monotonic_change")

    fp_score = -0.4 * len(fp)
    tp_score = 0.5 * len(tp)
    raw = tp_score + fp_score
    if catch.get("kind") == "behavior_diff":
        raw += 0.3
    score = max(-1.0, min(1.0, raw))
    return {"score": round(score, 3), "fp_patterns": fp, "tp_patterns": tp}
