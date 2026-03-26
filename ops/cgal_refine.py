# -*- coding: utf-8 -*-
"""Subprocess bridge for the CGAL checker and autorefinement executables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import os
import shlex
import shutil
import subprocess


class CGALBridgeError(RuntimeError):
    pass


class CGALBridgeBinaryMissing(CGALBridgeError):
    pass


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



def _candidate_executable_paths(build_dir: str | os.PathLike[str], name: str) -> List[Path]:
    build_dir = Path(build_dir)
    return [
        build_dir / name,
        build_dir / "Release" / name,
        build_dir / "Debug" / name,
        build_dir / "RelWithDebInfo" / name,
        build_dir / f"{name}.exe",
        build_dir / "Release" / f"{name}.exe",
        build_dir / "Debug" / f"{name}.exe",
        build_dir / "RelWithDebInfo" / f"{name}.exe",
    ]



def resolve_bridge_executable(build_dir: str | os.PathLike[str], name: str) -> Path:
    for candidate in _candidate_executable_paths(build_dir, name):
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    found_on_path = shutil.which(name)
    if found_on_path:
        return Path(found_on_path)
    raise CGALBridgeBinaryMissing(
        f"CGAL bridge executable '{name}' was not found under {build_dir!s}. "
        f"Build the bridge first with `cmake -S cgal_bridge -B build/cgal && cmake --build build/cgal -j`."
    )



def run_bridge_command(cmd: Sequence[str | os.PathLike[str]], timeout: int, cwd: str | os.PathLike[str] | None = None) -> CommandResult:
    cmd_strs = [str(c) for c in cmd]
    print(f"[CGAL] exec: {' '.join(shlex.quote(c) for c in cmd_strs)}")
    completed = subprocess.run(
        cmd_strs,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
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



def _parse_checker_output(stdout: str) -> tuple[bool, int, List[Tuple[int, int]]]:
    self_intersect: Optional[bool] = None
    count: Optional[int] = None
    pairs: List[Tuple[int, int]] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("self_intersect="):
            self_intersect = line.split("=", 1)[1].strip() == "1"
        elif line.startswith("count="):
            count = int(line.split("=", 1)[1].strip())
        elif line.startswith("pair="):
            rhs = line.split("=", 1)[1].strip()
            left, right = rhs.split(",", 1)
            pairs.append((int(left), int(right)))
    if self_intersect is None or count is None:
        raise CGALBridgeError(f"Unable to parse checker output:\n{stdout}")
    return self_intersect, count, pairs



def check_self_intersections(
    input_obj: str | os.PathLike[str],
    build_dir: str | os.PathLike[str] = "build/cgal",
    timeout: int = 60,
    list_pairs: bool = False,
) -> CheckResult:
    exe = resolve_bridge_executable(build_dir, "check_self_intersections")
    cmd: List[str | os.PathLike[str]] = [exe, input_obj]
    if list_pairs:
        cmd.append("--list_pairs")
    result = run_bridge_command(cmd, timeout=timeout)
    if result.returncode != 0:
        raise CGALBridgeError(
            f"CGAL checker failed with exit code {result.returncode}.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    self_intersect, count, pairs = _parse_checker_output(result.stdout)
    return CheckResult(self_intersect=self_intersect, count=count, pairs=pairs, command=result)



def autorefine_obj(
    input_obj: str | os.PathLike[str],
    output_obj: str | os.PathLike[str],
    build_dir: str | os.PathLike[str] = "build/cgal",
    timeout: int = 300,
    snap_grid_size: int = 23,
    number_of_iterations: int = 5,
) -> AutorefineResult:
    exe = resolve_bridge_executable(build_dir, "autorefine_obj")
    cmd: List[str | os.PathLike[str]] = [
        exe,
        input_obj,
        output_obj,
        "--snap_grid_size",
        str(int(snap_grid_size)),
        "--number_of_iterations",
        str(int(number_of_iterations)),
    ]
    result = run_bridge_command(cmd, timeout=timeout)
    if result.returncode != 0:
        raise CGALBridgeError(
            f"CGAL autorefine failed with exit code {result.returncode}.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    out_path = Path(output_obj)
    if not out_path.exists():
        raise CGALBridgeError(f"CGAL autorefine reported success but did not create {out_path}")
    return AutorefineResult(output_path=out_path, command=result)
