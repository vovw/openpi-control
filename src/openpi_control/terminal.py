"""Small Rich inference display; robot execution stays on the main thread."""

from __future__ import annotations

import argparse
import logging
import os
import select
import shlex
import sys
import threading
from collections import deque
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def styled_text(message: str) -> Text:
    """Style literal CLI text without interpreting user input as Rich markup."""
    text = Text(message, style=log_style(message))
    text.highlight_regex(r"(?m)^\S[^\n]*:\s*$", "bold")
    text.highlight_regex(r"(?<!\w)--?[a-zA-Z][a-zA-Z0-9-]*", "cyan")
    text.highlight_regex(r"\b(?:usage|options|positional arguments):", "bold yellow")
    text.highlight_regex(r"https?://[^\s]+", "underline blue")
    return text


class RichArgumentParser(argparse.ArgumentParser):
    """Keep argparse semantics and formatting, adding terminal-aware colors."""

    def _print_message(self, message, file=None):
        if message:
            Console(file=file or sys.stderr).print(styled_text(message), end="", soft_wrap=True)


class _StyledOutput:
    def __init__(self, stream):
        self.stream = stream
        self.console = Console(file=stream)

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def write(self, message):
        self.console.print(styled_text(message), end="", soft_wrap=True)
        return len(message)

    def flush(self):
        self.stream.flush()


@contextmanager
def styled_output(enabled=True):
    """Color ordinary commands without imposing a live screen or changing pipes."""
    with ExitStack() as stack:
        if enabled:
            if sys.stdout.isatty():
                stack.enter_context(redirect_stdout(_StyledOutput(sys.stdout)))
            if sys.stderr.isatty():
                stack.enter_context(redirect_stderr(_StyledOutput(sys.stderr)))
        yield


def log_style(message: str) -> str:
    lower = message.lower()
    if any(word in lower for word in ("error", "failed", "critical", "exception")):
        return "red"
    if any(word in lower for word in ("warning", "warn", "stalled")):
        return "yellow"
    if "all checks pass" in lower:
        return "green"
    return "default"


class LogBuffer:
    """Bounded lines, including partial writes, shared with the display thread."""

    def __init__(self) -> None:
        self.lines: deque[str] = deque(maxlen=500)
        self.lock = threading.Lock()

    def append(self, message: str) -> None:
        with self.lock:
            self.lines.extend(message.splitlines())

    def tail(self, count: int) -> list[str]:
        with self.lock:
            return list(self.lines)[-max(1, count) :]


class _Output:
    def __init__(self, buffer: LogBuffer, original: object) -> None:
        self.buffer = buffer
        self.original = original
        self.pending = ""
        self.lock = threading.RLock()

    def __getattr__(self, name: str) -> object:
        return getattr(self.original, name)

    def isatty(self) -> bool:
        return False

    def write(self, value: str) -> int:
        with self.lock:
            self.pending += value.replace("\r", "\n")
            while "\n" in self.pending or len(self.pending) > 4096:
                if "\n" in self.pending:
                    line, self.pending = self.pending.split("\n", 1)
                else:
                    line, self.pending = self.pending[:4096], self.pending[4096:]
                self.buffer.append(line)
                # Persist print() output as well as normal Python log records.
                logging.getLogger("openpi_control.terminal.output").info(line)
        return len(value)

    def flush(self) -> None:
        pass

    def finish(self) -> None:
        with self.lock:
            if self.pending:
                self.write("\n")


class _Logs(logging.Handler):
    def __init__(self, buffer: LogBuffer) -> None:
        super().__init__()
        self.buffer = buffer
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        if record.name != "openpi_control.terminal.output":
            self.buffer.append(self.format(record))


@contextmanager
def inference_terminal(args: object, stop: threading.Event, log_path: Path):
    """Use a dashboard only on an interactive POSIX terminal."""
    if (
        getattr(args, "plain", False)
        or os.name != "posix"
        or os.environ.get("TERM") == "dumb"
        or not all(stream.isatty() for stream in (sys.stdin, sys.stdout, sys.stderr))
    ):
        yield
        return

    import termios
    import tty

    console = Console(file=sys.stdout)
    buffer = LogBuffer()
    done = threading.Event()
    failed = threading.Event()
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    command = shlex.join(["openpi-control", *sys.argv[1:]]).replace("\n", " ").replace("\r", " ")

    def render() -> Group:
        command_line = Text("$ " + command, style="dim", overflow="ellipsis", no_wrap=True)
        instruction = str(args.instruction).replace("\n", " ").replace("\r", " ")
        prompt = Text("PROMPT  ", style="bold cyan")
        prompt.append(instruction, style="bold white")
        metadata = Table.grid(padding=(0, 3))
        metadata.add_column(overflow="fold")
        metadata.add_column(no_wrap=True)
        server = Text("SERVER  ", style="dim cyan")
        server.append(str(args.server), style="cyan")
        speed = Text("SPEED  ", style="dim magenta")
        speed.append(f"{args.speed:g}×", style="bold magenta")
        if getattr(args, "command", "infer") == "teleop":
            speed = Text(f"{args.backend.upper()}  ·  {args.rate:g} Hz", style="bold magenta")
        metadata.add_row(server, speed)
        header = Panel(
            Group(prompt, metadata, command_line),
            title=Text(
                f" openpi-control  /  {getattr(args, 'command', 'infer')} ", style="bold cyan"
            ),
            title_align="left",
            border_style="red" if failed.is_set() else "blue",
            padding=(0, 1),
        )
        header_height = len(console.render_lines(header, console.options))
        lines = Text("\n").join(
            Text(line, style=log_style(line), no_wrap=True, overflow="ellipsis")
            for line in buffer.tail(console.height - header_height - 2)
        )
        footer = Text(
            "Failed — q or Ctrl-C to close"
            if failed.is_set()
            else "Stopping — waiting for cleanup and parking…"
            if stop.is_set()
            else (
                "q stop & park  •  Ctrl-C interrupt"
                if getattr(args, "park", True)
                else "q stop (parking disabled)  •  Ctrl-C interrupt"
            ),
            style="red" if failed.is_set() else "yellow" if stop.is_set() else "dim",
        )
        return Group(
            header,
            lines,
            footer,
        )

    def show_error(error: Exception) -> None:
        logging.getLogger("openpi_control.terminal").error("%s", error)
        failed.set()
        live.update(render(), refresh=True)
        try:
            stop.wait()
        except KeyboardInterrupt:
            stop.set()

    root = logging.getLogger()
    handler = _Logs(buffer)
    # Existing console handlers may hold the original stderr and bypass redirects.
    console_handlers = [
        h
        for h in root.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        and h.stream in (sys.stdout, sys.stderr)
    ]
    with ExitStack() as stack:
        stack.callback(termios.tcsetattr, fd, termios.TCSADRAIN, previous)
        tty.setcbreak(fd)  # Keep SIGINT / Ctrl-C semantics intact.
        for h in console_handlers:
            root.removeHandler(h)
            stack.callback(root.addHandler, h)
        root.addHandler(handler)
        stack.callback(root.removeHandler, handler)
        live = stack.enter_context(
            Live(
                render(),
                console=console,
                screen=True,
                auto_refresh=False,
                redirect_stdout=False,
                redirect_stderr=False,
            )
        )
        outputs = [_Output(buffer, sys.stdout), _Output(buffer, sys.stderr)]
        stack.enter_context(redirect_stdout(outputs[0]))
        stack.enter_context(redirect_stderr(outputs[1]))

        def update() -> None:
            while not done.is_set():
                ready, _, _ = select.select([fd], [], [], 0.25)
                if ready and os.read(fd, 1) in (b"q", b"Q"):
                    stop.set()
                live.update(render(), refresh=True)

        thread = threading.Thread(target=update, name="terminal-display", daemon=True)
        thread.start()
        try:
            yield show_error
        finally:
            done.set()
            thread.join()
            for output in outputs:
                output.finish()
    # Alternate-screen history disappears on exit; retain a useful final summary.
    for line in buffer.tail(8):
        console.print(Text(line, style=log_style(line)))
    console.print(f"Full log: {log_path}", markup=False)
