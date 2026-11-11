"""Replay threads that need the engine's image layout at the same time all get the whole module."""
from concurrent.futures import ThreadPoolExecutor
import sys
import threading
import time

from trie import engine_files


def test_concurrent_loads_wait_for_the_module_to_finish(tmp_path):
    source = tmp_path / "slow_layout.py"
    source.write_text("import time\ntime.sleep(0.2)\ndef prepare_image_layout():\n    return 'ready'\n")
    name = "trie_test_slow_layout"
    sys.modules.pop(name, None)
    start = threading.Barrier(8)

    def load():
        start.wait()
        return engine_files.load_file(source, name).prepare_image_layout()

    try:
        with ThreadPoolExecutor(8) as pool:
            assert [f.result() for f in [pool.submit(load) for _ in range(8)]] == ["ready"] * 8
    finally:
        sys.modules.pop(name, None)
