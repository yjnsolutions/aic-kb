from aic_kb.cli import build_string


def test_build_string():
    build_string("world", 3) == "worldworldworld"
