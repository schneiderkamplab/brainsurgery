"""
Example:
python validation/count_lines.py validation/pytorch_example.py validation/validation.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def _strip_python_style_comments(
	line: str,
	in_block_comment: str | None,
) -> tuple[str, str | None]:
	"""Strip python-style comments from a single line.

	Supports:
	- Single-line comments beginning with '#'.
	- Multi-line comments enclosed by triple quotes (double or single).
	"""
	i = 0
	chars: list[str] = []
	while i < len(line):
		if in_block_comment is not None:
			end = line.find(in_block_comment, i)
			if end == -1:
				return "".join(chars), in_block_comment
			i = end + 3
			in_block_comment = None
			continue

		if line.startswith('"""', i):
			in_block_comment = '"""'
			i += 3
			continue

		if line.startswith("'''", i):
			in_block_comment = "'''"
			i += 3
			continue

		if line[i] == "#":
			break

		chars.append(line[i])
		i += 1

	return "".join(chars), in_block_comment


def _count_file_code_lines(file_path: Path) -> int:
	count = 0
	in_block_comment: str | None = None

	with file_path.open("r", encoding="utf-8", errors="replace") as handle:
		for raw_line in handle:
			line = raw_line.rstrip("\n")
			stripped, in_block_comment = _strip_python_style_comments(line, in_block_comment)
			if stripped.strip():
				count += 1

	return count


def _expand_input_paths(paths: Iterable[Path]) -> list[Path]:
	files: list[Path] = []
	for path in paths:
		candidate = path.expanduser()
		if candidate.is_file():
			files.append(candidate)
		elif candidate.is_dir():
			files.extend(sorted(p for p in candidate.rglob("*") if p.is_file()))
		else:
			raise FileNotFoundError(f"Path does not exist: {candidate}")

	return files


def count_lines(paths: Iterable[str | Path]) -> dict[Path, int]:
	"""Count code lines for one or more files.

	The count excludes blank lines and python-style comments.
	"""
	path_objects = [Path(path) for path in paths]
	files = _expand_input_paths(path_objects)
	return {file_path: _count_file_code_lines(file_path) for file_path in files}


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Count code lines excluding blank lines and python-style comments.",
	)
	parser.add_argument("paths", nargs="+", help="One or more files (or directories) to count")
	args = parser.parse_args()

	try:
		results = count_lines(args.paths)
	except FileNotFoundError as error:
		parser.error(str(error))

	for file_path, line_count in results.items():
		print(f"{file_path}: {line_count}")

	if len(results) > 1:
		print(f"TOTAL: {sum(results.values())}")


if __name__ == "__main__":
	main()
