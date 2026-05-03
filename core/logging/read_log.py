from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union


PathLike = Union[str, Path]


class LogReader:
    """
    Read one SimpleLogger2 table log and expose columns with ``[]`` indexing.

    Example:
        reader = LogReader("output/run/log.txt")
        losses = reader["loss"]
        first_row = reader[0]

    Read train rows from a SimpleLogger2 run directory:
        reader = LogReader("output/run", channel="T")
        steps = reader["global_step"]
        losses = reader["loss"]
        psnr = reader["psnr"]

    Use rows when per-step records are more convenient:
        for row in reader:
            step = row["global_step"]
            loss = row["loss"]

    Use a column dictionary when comparing multiple metrics:
        columns = reader.to_columns()
        loss = columns["loss"]
        mae = columns["mae"]
        mse = columns["mse"]

    Export every column to two-column text files:
        output_dir = reader.export_columns()

    Generate text files directly from the command line:
        python core/logging/read_log.py --path output/run/log.txt
        python core/logging/read_log.py --path output/run/train.log --channel L
    """

    def __init__(
        self,
        path: PathLike,
        *,
        channel: Optional[str] = None,
        cast_values: bool = True,
        value_width: Optional[int] = None,
        strict: bool = False,
        encoding: str = "utf-8",
    ):
        self.path = self._resolve_path(path, channel)
        self.channel = channel
        self.cast_values = cast_values
        self.value_width = value_width
        self.strict = strict
        self.encoding = encoding
        self.header: List[str] = []
        self.rows: List[Dict[str, Any]] = []
        self.read()

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            if self.header and key not in self.header:
                raise KeyError(key)
            return [row.get(key, "") for row in self.rows]
        if isinstance(key, (int, slice)):
            return self.rows[key]
        raise TypeError("LogReader indices must be a column name, int, or slice.")

    @property
    def columns(self) -> List[str]:
        return list(self.header)

    def to_rows(self) -> List[Dict[str, Any]]:
        return list(self.rows)

    def to_columns(self) -> Dict[str, List[Any]]:
        return {column: self[column] for column in self.header}

    def export_columns(
        self,
        output_dir: Optional[PathLike] = None,
        *,
        index_column: str = "log_index",
        skip_index_column: bool = False,
    ) -> Path:
        """
        Write each data column into a separate two-column text file.

        The output format has no header. Each line is:
            log_index value
        """
        if output_dir is None:
            output_dir = self.path.with_suffix("").parent / f"{self.path.stem}_columns"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        indices = self[index_column] if index_column in self.header else list(range(len(self)))
        for column in self.header:
            if skip_index_column and column == index_column:
                continue
            column_path = output_path / f"{self._safe_column_name(column)}.txt"
            with column_path.open("w", encoding=self.encoding) as f:
                for log_index, value in zip(indices, self[column]):
                    f.write(f"{self._format_output_value(log_index)} {self._format_output_value(value)}\n")

        return output_path

    def read(self) -> "LogReader":
        with self.path.open("r", encoding=self.encoding) as f:
            self.header, self.rows = self._parse_lines(f)
        return self

    def _parse_lines(self, lines: Iterable[str]):
        header: List[str] = []
        rows: List[Dict[str, Any]] = []

        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.rstrip("\n")
            if line.startswith("[H] "):
                header = line[4:].split()
                continue

            if not self._is_data_line(line):
                continue
            line_prefix = line[1]
            if self.channel and line_prefix != self.channel:
                continue

            if not header:
                if self.strict:
                    raise ValueError(f"Data row before [H] header at line {line_number}.")
                continue

            values = self._split_values(line[4:], len(header))
            if len(values) != len(header):
                if self.strict:
                    raise ValueError(
                        f"Expected {len(header)} values at line {line_number}, "
                        f"got {len(values)}."
                    )
                values = self._fit_values(values, len(header))

            if self.cast_values:
                row = {key: self.parse_value(value) for key, value in zip(header, values)}
            else:
                row = {key: value.strip() for key, value in zip(header, values)}
            rows.append(row)

        return header, rows

    def _split_values(self, payload: str, expected_count: int) -> List[str]:
        if self.value_width is not None:
            if self.value_width < 1:
                raise ValueError("value_width must be >= 1.")
            return self._split_fixed_width_values(payload, expected_count)

        if expected_count <= 1:
            return [payload.strip()]
        return payload.split(maxsplit=expected_count - 1)

    def _split_fixed_width_values(self, payload: str, expected_count: int) -> List[str]:
        values = []
        start = 0
        for index in range(expected_count):
            if index == expected_count - 1:
                values.append(payload[start:].strip())
                break
            values.append(payload[start : start + self.value_width].strip())
            start += self.value_width + 1
        return values

    @staticmethod
    def _fit_values(values: Sequence[str], expected_count: int) -> List[str]:
        if len(values) < expected_count:
            return [*values, *([""] * (expected_count - len(values)))]
        if len(values) > expected_count:
            return [*values[: expected_count - 1], " ".join(values[expected_count - 1 :])]
        return list(values)

    @staticmethod
    def parse_value(value: str) -> Any:
        value = value.strip()
        if value == "":
            return ""

        lower = value.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
        if lower == "none":
            return None
        if lower == "nan":
            return math.nan
        if lower == "inf":
            return math.inf
        if lower == "-inf":
            return -math.inf

        if re.fullmatch(r"[+-]?\d+", value):
            try:
                return int(value)
            except ValueError:
                return value

        if re.fullmatch(
            r"[+-]?((\d+\.\d*)|(\d*\.\d+)|(\d+))([eE][+-]?\d+)?",
            value,
        ):
            try:
                return float(value)
            except ValueError:
                return value

        return value

    @staticmethod
    def _is_data_line(line: str) -> bool:
        return (
            len(line) > 4
            and line[0] == "["
            and line[2:4] == "] "
            and line[1] not in {"H", "I", "E"}
        )

    @staticmethod
    def _resolve_path(path: PathLike, channel: Optional[str]) -> Path:
        log_path = Path(path)
        if log_path.is_dir():
            log_path = log_path / "log.txt"
        return log_path

    @staticmethod
    def _safe_column_name(column: str) -> str:
        safe_name = re.sub(r'[<>:"/\\|?*\s]+', "_", column.strip())
        return safe_name.strip("._") or "column"

    @staticmethod
    def _format_output_value(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.12g}"
        return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export SimpleLogger2 table columns into two-column text files."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Explicit log file path, e.g. log.txt or legacy train.log.",
    )
    parser.add_argument(
        "--channel",
        default="T",
        help="One-character row prefix to export. Default: T.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Default: <log_stem>_columns beside the log file.",
    )
    parser.add_argument(
        "--value_width",
        default=None,
        type=int,
        help="Optional fixed width for parsing [L] values, e.g. 10.",
    )
    parser.add_argument(
        "--no_cast",
        action="store_true",
        help="Keep all parsed values as strings.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise errors for malformed table rows.",
    )
    parser.add_argument(
        "--skip_log_index",
        action="store_true",
        help="Do not export log_index.txt. It is exported by default.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Path:
    args = build_parser().parse_args(argv)
    log_path = Path(args.path)
    if not log_path.is_file():
        raise FileNotFoundError(f"--path must point to a log file: {log_path}")

    reader = LogReader(
        log_path,
        channel=args.channel,
        cast_values=not args.no_cast,
        value_width=args.value_width,
        strict=args.strict,
    )
    output_dir = reader.export_columns(
        args.output_dir,
        skip_index_column=args.skip_log_index,
    )
    print(output_dir)
    return output_dir


__all__ = ["LogReader", "build_parser", "main"]


if __name__ == "__main__":
    main()
