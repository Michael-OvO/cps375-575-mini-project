from pipeline.constraints import ifevalg
from pipeline.generate import if_rlvr


def test_ifevalg_checks_new_and_standard_types():
    ids = ["last_word:last_word_answer", "punctuation:no_comma"]
    kw = [{"last_word": "brief"}, None]
    assert ifevalg.passes("Keep it short and brief", ids, kw)
    assert not ifevalg.passes("Keep it short, and brief", ids, kw)  # comma
    assert not ifevalg.passes("Keep it brief and short", ids, kw)  # wrong last word
    assert not ifevalg.passes("   ", ["punctuation:no_comma"], [None])  # empty never passes
    assert ifevalg.check("one two", ["punctuation:punctuation_dot"], [None]) == [True]
    assert ifevalg.check("one. two", ["punctuation:punctuation_dot"], [None]) == [False]


def test_parse_ground_truth_reads_dataset_format():
    gt = "[{'instruction_id': ['detectable_format:sentence_hyphens', 'last_word:last_word_answer'], " \
         "'kwargs': [None, {'last_word': 'brief'}]}]"
    assert ifevalg.parse_ground_truth(gt) == (["detectable_format:sentence_hyphens", "last_word:last_word_answer"],
                                              [None, {"last_word": "brief"}])


def test_base_task_strips_constraints_or_refuses():
    prompt = "Name two string instruments. Use no commas. End with the word brief."
    assert if_rlvr.base_task(prompt, "Use no commas.\tEnd with the word brief.") == "Name two string instruments."
    assert if_rlvr.base_task(prompt, "A constraint that is not there.") is None


def test_near_dup_catches_rewordings_not_different_tasks():
    nd = if_rlvr.NearDup()
    a = "Write a short essay about how community gardens bring neighbors together in dense city neighborhoods today"
    assert nd.add(a)
    assert not nd.add(a + " please")  # near-identical
    assert nd.add("Explain how a binary search tree keeps its keys ordered and why lookups take log time")


def test_held_out_types_are_real_and_not_standard():
    assert if_rlvr.HELD_OUT <= set(ifevalg.REGISTRY)
    assert len(ifevalg.REGISTRY) == 54


def test_respond_retries_with_feedback_and_keeps_original_prompt(monkeypatch):
    import asyncio
    seen = []

    async def fake(prompt, system=None, max_tokens=0, effort="medium"):
        seen.append(prompt)
        return "Short, and brief" if len(seen) == 1 else "Short and brief"

    monkeypatch.setattr(if_rlvr.teachers, "get", lambda name: fake)
    item = {"id": "rl-1", "task": "t", "prompt": "Say something.", "source": "s", "new_types": [],
            "constraints": [{"id": "punctuation:no_comma", "kwargs": None}]}
    rows = asyncio.run(if_rlvr.respond(item))
    assert rows[0]["attempts"] == 2 and rows[0]["prompt"] == "Say something."
    assert "refrain from the use of any commas" in seen[1] and "Short, and brief" in seen[1]
