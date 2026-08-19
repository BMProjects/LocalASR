"""One desktop host per user session.

Three windows must not become three processes: each would build its own EngineManager
and try to start its own llama-server, which on a 4 GB card means the second one fails
to allocate. A second launch instead asks the running host to show the window that was
requested, and exits.

The running host must *acknowledge* the request. Without an ack, a host that is hung —
or one left over from an older build that does not know how to show the requested
window — still accepts the connection, and every subsequent launch exits silently
having done nothing. From the user's side the application simply stops starting.

Uses an abstract AF_UNIX socket, so there is no stale file to clean up after a crash:
the address is released with the process.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable

ADDRESS = "\0localasr-desktop"
ACK = b"ok"
QUIT = "quit"


class AlreadyRunning(RuntimeError):
    """Another host owns the address and acknowledged the message."""


class SingleInstance:
    """Owns the address, or reports who does."""

    def __init__(self, address: str = ADDRESS) -> None:
        self.address = address
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def acquire(self, on_message: Callable[[str], None]) -> None:
        """Claim the address and serve activation requests, or raise AlreadyRunning."""
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(self.address)
        except OSError as exc:
            server.close()
            raise AlreadyRunning(f"another localasr desktop is running: {exc}") from exc

        server.listen(4)
        server.settimeout(0.5)
        self._socket = server
        self._thread = threading.Thread(
            target=self._serve, args=(on_message,), name="localasr-instance", daemon=True
        )
        self._thread.start()

    def _serve(self, on_message: Callable[[str], None]) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._socket.accept()
            except (TimeoutError, OSError):
                continue
            with conn:
                try:
                    payload = conn.recv(4096).decode("utf-8", "replace").strip()
                    if payload:
                        # Acknowledge before handling: the caller only needs to know
                        # that a live host received the request.
                        conn.sendall(ACK)
                except OSError:
                    continue
            if payload:
                on_message(payload)

    def release(self) -> None:
        self._stop.set()
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def __enter__(self) -> SingleInstance:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def activate(message: str, address: str = ADDRESS, timeout: float = 3.0) -> bool:
    """Ask a running host to act on `message`.

    True only when a host answered. A host that connects but never acknowledges is
    treated as absent, so the caller starts its own rather than exiting into silence.
    """
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout)
    try:
        client.connect(address)
        client.sendall(message.encode("utf-8"))
        return client.recv(len(ACK)) == ACK
    except OSError:
        return False
    finally:
        client.close()


def request_quit(address: str = ADDRESS, timeout: float = 5.0) -> bool:
    """Ask a running host to exit. False when there was none to ask."""
    return activate(QUIT, address, timeout)
