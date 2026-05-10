# -*- coding: utf-8 -*-
"""CGAL 检测器和自交修复程序的 subprocess 桥接层。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import os
import shlex
import shutil
import subprocess


PathLike = str | os.PathLike[str]


class CGALBridgeError(RuntimeError):
    """CGAL 桥接层统一异常。"""


class CGALBridgeBinaryMissing(CGALBridgeError):
    """找不到 CGAL 桥接可执行文件。"""


@dataclass
class CommandResult:
    cmd: List[str]
    returncode: int
    stdout: str
    stderr: str


@dataclass
class CheckResult:
    self_intersect: bool
    count: int
    pairs: List[Tuple[int, int]]
    command: CommandResult


@dataclass
class AutorefineResult:
    output_path: Path
    command: CommandResult


def _candidate_executable_paths(build_dir: PathLike, name: str) -> List[Path]:
    """列出常见 CMake 构建目录下的可执行文件位置。"""
    root = Path(build_dir)
    return [
        root / name,
        root / "Release" / name,
        root / "Debug" / name,
        root / "RelWithDebInfo" / name,
        root / f"{name}.exe",
        root / "Release" / f"{name}.exe",
        root / "Debug" / f"{name}.exe",
        root / "RelWithDebInfo" / f"{name}.exe",
    ]


def _is_executable(path: Path) -> bool:
    """判断路径是否是可执行文件。"""
    return path.is_file() and os.access(path, os.X_OK)


def _format_cmd(cmd: Sequence[PathLike]) -> str:
    return " ".join(shlex.quote(str(c)) for c in cmd)


def _ensure_input_file(path: PathLike, label: str) -> Path:
    p = Path(path)
    if not p.is_file():
        raise CGALBridgeError(f"{label} 不存在或不是文件: {p}")
    return p


def _ensure_positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} 必须是正整数，当前为 {value}")
    return value


def resolve_bridge_executable(build_dir: PathLike, name: str) -> Path:
    """在 build_dir 和 PATH 中查找 CGAL 桥接可执行文件。"""
    for candidate in _candidate_executable_paths(build_dir, name):
        if _is_executable(candidate):
            return candidate

    found_on_path = shutil.which(name)
    if found_on_path:
        return Path(found_on_path)

    raise CGALBridgeBinaryMissing(
        f"找不到 CGAL 桥接可执行文件: {name}\n"
        f"已检查目录: {Path(build_dir)}\n"
        f"请先编译:\n"
        f"cmake -S cgal_bridge -B build/cgal\n"
        f"cmake --build build/cgal -j"
    )


def run_bridge_command(
    cmd: Sequence[PathLike],
    timeout: int,
    cwd: PathLike | None = None,
) -> CommandResult:
    """执行 CGAL 桥接命令，并保留 stdout/stderr 方便上层报错。"""
    timeout = _ensure_positive_int(timeout, "timeout")
    cmd_strs = [str(c) for c in cmd]

    print(f"[CGAL] exec: {_format_cmd(cmd_strs)}")

    try:
        completed = subprocess.run(
            cmd_strs,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CGALBridgeError(
            f"CGAL 命令超时，timeout={timeout}s:\n{_format_cmd(cmd_strs)}"
        ) from exc
    except OSError as exc:
        raise CGALBridgeError(
            f"CGAL 命令启动失败:\n{_format_cmd(cmd_strs)}"
        ) from exc

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()

    if stdout:
        print("[CGAL][stdout]")
        print(stdout)
    if stderr:
        print("[CGAL][stderr]")
        print(stderr)

    return CommandResult(
        cmd=cmd_strs,
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _parse_bool_flag(value: str) -> bool:
    value = value.strip()
    if value == "1":
        return True
    if value == "0":
        return False
    raise ValueError(f"布尔标记必须是 0 或 1，当前为 {value!r}")


def _parse_checker_output(stdout: str) -> tuple[bool, int, List[Tuple[int, int]]]:
    """解析 check_self_intersections 的文本输出。"""
    self_intersect: Optional[bool] = None
    count: Optional[int] = None
    pairs: List[Tuple[int, int]] = []

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        key, sep, value = line.partition("=")
        if not sep:
            continue

        key = key.strip()
        value = value.strip()

        try:
            if key == "self_intersect":
                self_intersect = _parse_bool_flag(value)
            elif key == "count":
                count = int(value)
            elif key == "pair":
                left, right = value.split(",", 1)
                pairs.append((int(left), int(right)))
        except ValueError as exc:
            raise CGALBridgeError(f"无法解析 CGAL checker 输出行: {line!r}") from exc

    if self_intersect is None or count is None:
        raise CGALBridgeError(f"CGAL checker 输出缺少必要字段:\n{stdout}")

    return self_intersect, count, pairs


def check_self_intersections(
    input_obj: PathLike,
    build_dir: PathLike = "build/cgal",
    timeout: int = 60,
    list_pairs: bool = False,
) -> CheckResult:
    """调用 CGAL 检测 OBJ 是否存在自交。"""
    input_path = _ensure_input_file(input_obj, "输入 OBJ")
    exe = resolve_bridge_executable(build_dir, "check_self_intersections")

    cmd: List[PathLike] = [exe, input_path]
    if list_pairs:
        cmd.append("--list_pairs")

    result = run_bridge_command(cmd, timeout=timeout)

    if result.returncode != 0:
        raise CGALBridgeError(
            f"CGAL 自交检测失败，退出码={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    self_intersect, count, pairs = _parse_checker_output(result.stdout)
    return CheckResult(
        self_intersect=self_intersect,
        count=count,
        pairs=pairs,
        command=result,
    )


def autorefine_obj(
    input_obj: PathLike,
    output_obj: PathLike,
    build_dir: PathLike = "build/cgal",
    timeout: int = 300,
    snap_grid_size: int = 23,
    number_of_iterations: int = 5,
) -> AutorefineResult:
    """调用 CGAL autorefine，输出修复后的 OBJ。"""
    input_path = _ensure_input_file(input_obj, "输入 OBJ")
    output_path = Path(output_obj)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    snap_grid_size = _ensure_positive_int(snap_grid_size, "snap_grid_size")
    number_of_iterations = _ensure_positive_int(number_of_iterations, "number_of_iterations")

    exe = resolve_bridge_executable(build_dir, "autorefine_obj")

    cmd: List[PathLike] = [
        exe,
        input_path,
        output_path,
        "--snap_grid_size",
        str(snap_grid_size),
        "--number_of_iterations",
        str(number_of_iterations),
    ]

    result = run_bridge_command(cmd, timeout=timeout)

    if result.returncode != 0:
        raise CGALBridgeError(
            f"CGAL autorefine 失败，退出码={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    if not output_path.is_file():
        raise CGALBridgeError(f"CGAL autorefine 返回成功，但没有生成输出文件: {output_path}")

    return AutorefineResult(output_path=output_path, command=result)