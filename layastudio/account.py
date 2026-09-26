"""Sign in to System One Models from the studio's page, with the CLI's own login.

The page shows a popup with a one-time code; the registry's approval page opens in a new
tab; the person approves; the studio saves the token exactly where `systemone login`
saves it. It *is* the CLI's login — the same device-authorization flow from the
`systemone` package, the same config file — so signing in here signs in the terminal
too, and the other way round. Signing out runs `systemone logout`, which also revokes the
token on the registry.

The token never reaches the page: the page only learns the username.
"""

import os
import socket
import subprocess
import threading
import time


def _systemone():
    try:
        from systemone import config
        from systemone.auth import device_login
        from systemone.client import Client
        from systemone.errors import SystemOneError
    except ImportError:
        return None
    return config, device_login, Client, SystemOneError


class Account:
    """One sign-in at a time, run in the background while the page polls."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pending = None  # the codes the page shows while waiting
        self._error = None
        self._cancel = threading.Event()
        self._thread = None
        self._checked = {}  # token prefix -> (checked_at, username or None)

    def status(self):
        modules = _systemone()
        if modules is None:
            return {
                "available": False,
                "signed_in": False,
                "detail": "The systemone package is not installed. Start the studio with "
                "`systemone run studio`, or install it: pip install systemonemodels",
            }
        config = modules[0]
        stored = config.load()
        with self._lock:
            pending, error = self._pending, self._error
        return {
            "available": True,
            "signed_in": bool(stored.token),
            "username": stored.username,
            "site": config.web_url(stored.endpoint),
            "from_environment": bool(os.environ.get(config.ENV_TOKEN)),
            "pending": pending,
            "error": error,
        }

    def start(self):
        modules = _systemone()
        if modules is None:
            raise RuntimeError("The systemone package is not installed")
        config, device_login, Client, SystemOneError = modules
        with self._lock:
            if self._thread and self._thread.is_alive():
                return self.status()
            self._pending, self._error = None, None
            self._cancel.clear()

        def on_code(codes):
            with self._lock:
                self._pending = {
                    "user_code": codes["user_code"],
                    "verification_uri": codes["verification_uri"],
                    "verification_uri_complete": codes["verification_uri_complete"],
                    "expires_in": codes.get("expires_in"),
                }

        def sleep(seconds):
            # The CLI's poller sleeps between polls; a cancel from the page ends it here.
            if self._cancel.wait(seconds):
                raise InterruptedError("Sign-in cancelled")

        def run():
            current = config.load(environment=False)
            current.token = None
            try:
                with Client(current) as anonymous:
                    granted = device_login(
                        anonymous,
                        f"Laya Studio on {socket.gethostname() or 'this machine'}",
                        on_code,
                        sleep=sleep,
                    )
                current.token = granted["token"]
                current.username = str(granted["username"])
                config.save(current)
                with self._lock:
                    self._pending = None
            except InterruptedError:
                with self._lock:
                    self._pending = None
            except (SystemOneError, OSError) as error:
                with self._lock:
                    self._pending, self._error = None, str(error)

        self._thread = threading.Thread(target=run, name="systemone-login", daemon=True)
        self._thread.start()
        # Give the registry a moment to hand out the codes, so the first answer has them.
        for _ in range(30):
            with self._lock:
                if self._pending or self._error:
                    break
            time.sleep(0.1)
        return self.status()

    def cancel(self):
        self._cancel.set()
        return self.status()

    def logout(self):
        """`systemone logout`: revoke the token on the registry and forget it here."""
        from .publish_systemone import cli_command

        done = subprocess.run(
            [*cli_command(), "logout"], capture_output=True, text=True, timeout=60, check=False
        )
        status = self.status()
        status["message"] = (done.stdout or done.stderr).strip().splitlines()[-1:] or [""]
        status["message"] = status["message"][0]
        return status
