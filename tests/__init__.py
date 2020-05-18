from pathlib import Path

TEST_FILES_PATH = Path(__file__).parent / "test_files"

print("Loading test data...")

TEST_FILES = {x.name: x for x in TEST_FILES_PATH.iterdir()}

print("Finished loading test files")
