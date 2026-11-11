"""tools/scrub_transcript.py: what it replaces, what it flags, what it leaves."""
import importlib.util
import json
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "tools/scrub_transcript.py"
spec = importlib.util.spec_from_file_location("scrub_transcript", PATH)
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)

DOCKER = "dckr_pat_" + "Ab3dEf6hIj9kLmN2pQr5sTu8vWx"
HF = "hf_" + "aBcDeFgHiJkLmNoPqRsTuVwXyZ012345"


def test_tokens_and_private_data_are_replaced():
    text = (f"here is my token {DOCKER} and {HF}\nexport HF_TOKEN={HF}\n"
            "curl -H 'Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345' https://user:pa55word@host/x\n"
            "mail zdu@example.com, file /Users/alice/repo/a.py, node 10.1.2.3, local 127.0.0.1\n"
            "https://s3/x?X-Amz-Signature=0123456789abcdef0123456789abcdef&X-Amz-Date=20261110T000000Z")
    out, counts, left = st.scrub(text)
    for secret in (DOCKER, HF, "abcdefghijklmnopqrstuvwxyz012345", "pa55word", "zdu@example.com", "alice",
                   "10.1.2.3", "0123456789abcdef0123456789abcdef"):
        assert secret not in out
    assert "127.0.0.1" in out and "/Users/<user>/repo/a.py" in out and "HF_TOKEN=[REDACTED]" in out
    assert counts["docker token"] == 1 and counts["email"] == 1 and not left
    assert "X-Amz-Date=20261110T000000Z" in out


def test_ordinary_text_stays():
    text = ("max_tokens=1024 role_control_token_file=/secure/role-token tokenizer_model=/models/GLM-5.2-NVFP4\n"
            "sha256 b51e7b38bbeda911833d18abc846368b3530a3c5c6c134d1552973eb65f2682a\n"
            "DeepSeek-V4-Flash-Vision-Exp-6821d6ad3681a4b137b066b76094fa82ebd0a380\n"
            "--symmetric-pd-external-role-control SGLANG_DSA_HOST_DEDUP_STREAM_REDRAW=1\n"
            "Decode batch, #token: 12345678, cached-token=87654321012345678 X-Amz-Algorithm=AWS4-HMAC-SHA256")
    out, counts, left = st.scrub(text)
    assert out == text and not counts and not left


def test_an_unknown_secret_is_flagged_not_published(tmp_path):
    secret = "Zq7Lm2Xv9Kp4Rt6Yw1Bn8Cd3Fg5Hj0Ms"
    session = tmp_path / "s.txt"
    session.write_text(f"the value {secret} was pasted")
    assert st.main([str(session), "-o", str(tmp_path / "out.md")]) == 1
    assert st.scrub(session.read_text())[2] == [(1, secret)]
    assert st.main([str(session), "-o", str(tmp_path / "out.md"), "--allow", secret[:16]]) == 0


def test_names_given_with_redact_are_replaced_as_whole_words():
    out, counts, _ = st.scrub("/scratch/users/alice/run on node-17, alicent stays", redact=("alice", "node-17"))
    assert out == "/scratch/users/<redacted>/run on <redacted>, alicent stays" and counts["name"] == 2


def test_claude_code_session_renders_messages_tools_and_results(tmp_path):
    rows = [
        {"type": "user", "message": {"content": "fix the bug"}},
        {"type": "assistant", "message": {"content": [
            {"type": "thinking", "thinking": "hidden"},
            {"type": "text", "text": "Looking."},
            {"type": "tool_use", "name": "Bash", "input": {"command": "pytest -q"}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "content": [{"type": "text", "text": "3 passed"}]}]}},
        {"type": "assistant", "isSidechain": True, "message": {"content": "subagent"}},
        {"type": "system", "content": "ignored"},
    ]
    session = tmp_path / "s.jsonl"
    session.write_text("\n".join(json.dumps(r) for r in rows))
    out = st.render(session)
    assert "fix the bug" in out and "**Tool: Bash**" in out and "pytest -q" in out and "3 passed" in out
    assert "hidden" not in out and "subagent" not in out and "ignored" not in out
