import io
import os
import select
import subprocess
import sys
import threading
import time
from argparse import Namespace

import pytest

from openpi_control.terminal import LogBuffer, _Output, inference_terminal, log_style


def test_output_joins_partial_writes_and_bounds_history():
    buffer = LogBuffer()
    output = _Output(buffer, io.StringIO())
    output.write("preflight ")
    output.write("left: all checks pass\n")
    assert buffer.tail(1) == ["preflight left: all checks pass"]
    for index in range(600):
        output.write(f"{index}\n")
    assert len(buffer.lines) == 500
    assert buffer.tail(2) == ["598", "599"]
    output.write("partial")
    output.finish()
    assert buffer.tail(1) == ["partial"]


def test_plain_does_not_capture_output_or_request_stop(capsys, tmp_path):
    stop = threading.Event()
    with inference_terminal(Namespace(plain=True), stop, tmp_path / "infer.log"):
        print("still plain")
    assert capsys.readouterr().out == "still plain\n"
    assert not stop.is_set()


def test_severity_colors():
    assert log_style("inference stopped: InferenceError: offline") == "red"
    assert log_style("WARNING stale frame") == "yellow"
    assert log_style("preflight left: all checks pass") == "green"


def test_help_is_colored_in_terminal_and_plain_in_pipe(monkeypatch):
    from rich.console import Console

    from openpi_control import terminal

    parser = terminal.RichArgumentParser(prog="control")
    sub = parser.add_subparsers().add_parser("infer")
    sub.add_argument("--server", help="server address")
    plain = io.StringIO()
    sub.print_help(plain)
    assert "--server" in plain.getvalue()
    assert "\x1b[" not in plain.getvalue()

    monkeypatch.setattr(
        terminal, "Console", lambda **kw: Console(force_terminal=True, **kw)
    )
    colored = io.StringIO()
    sub.print_help(colored)
    assert "\x1b[" in colored.getvalue()
    assert "--server" in colored.getvalue()


@pytest.mark.skipif(os.name != "posix", reason="POSIX terminal integration")
@pytest.mark.parametrize("mode", ["running", "error"])
def test_q_stops_and_restores_terminal(tmp_path, mode):
    import pty
    import termios

    master, slave = pty.openpty()
    before = termios.tcgetattr(slave)
    code = """
from argparse import Namespace
from pathlib import Path
from threading import Event
import sys
from openpi_control.terminal import inference_terminal
stop = Event()
args = Namespace(plain=False, instruction='fold [red]literal', server='test', speed=0.5)
with inference_terminal(args, stop, Path('test.log')) as show_error:
    if sys.argv[1] == 'error':
        show_error(RuntimeError('READY connection refused'))
    else:
        print('READY')
        assert stop.wait(8), 'q did not request stop'
print('CLEAN EXIT')
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, mode],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env={**os.environ, "TERM": "xterm-256color"},
    )
    output = b""
    try:
        deadline = time.monotonic() + 6
        while b"READY" not in output and time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                output += os.read(master, 65536)
        assert b"READY" in output, output
        assert process.poll() is None
        if mode == "error":
            assert b"Failed" in output
        os.write(master, b"q")
        assert process.wait(timeout=4) == 0
        while select.select([master], [], [], 0.1)[0]:
            output += os.read(master, 65536)
        assert b"CLEAN EXIT" in output
        assert b"\x1b[?1049h" in output  # Enter and leave alternate screen.
        assert b"\x1b[?1049l" in output
        assert termios.tcgetattr(slave) == before
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)
        os.close(slave)
