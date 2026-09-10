"""System-audio capture selection, readiness checks and desktop launchers."""

import subprocess

import numpy as np
import pytest

from localasr.capture import pulse
from localasr.frontends.cli import doctor
from localasr.frontends.desktop import entries

# Trimmed from real `pactl list short sources` output on this machine.
PACTL_SOURCES = "\n".join(
    [
        "56\talsa_output.pci-0000_00_1f.3.HiFi__HDMI1__sink.monitor\tPipeWire\tSUSPENDED",
        "60\talsa_input.pci-0000_00_1f.3.HiFi__Mic2__source\tPipeWire\tSUSPENDED",
        "62\tbluez_output.00_42_79_F4_56_6D.1.monitor\tPipeWire\tRUNNING",
    ]
)


def _fake_sounddevice(rate: float, *, accepts: set[int], opened: dict):
    """A device that behaves like a raw ALSA PCM: one rate, everything else refused."""

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def read(self, frames):
            return np.full((frames, 1), 0.01, dtype=np.float32), False

    class FakeSoundDevice:
        @staticmethod
        def query_devices(_index=None):
            return {"name": "fake", "default_samplerate": rate, "max_input_channels": 2}

        @staticmethod
        def check_input_settings(*, samplerate, **_kwargs):
            if int(samplerate) not in accepts:
                raise RuntimeError("Invalid sample rate [PaErrorCode -9997]")

        @staticmethod
        def InputStream(**kwargs):
            opened.update(kwargs)
            return FakeStream()

    return FakeSoundDevice()


def test_input_probe_reports_a_measured_level(monkeypatch):
    from localasr.capture import microphone

    opened = {}
    monkeypatch.setattr(
        microphone,
        "_sounddevice",
        lambda: _fake_sounddevice(16_000.0, accepts={16_000}, opened=opened),
    )

    level = microphone.probe_input(5, duration=0.1)

    assert opened["samplerate"] == 16_000
    assert level.rms == pytest.approx(0.01)
    assert level.dbfs == pytest.approx(-40.0)
    assert not level.overflowed


def test_a_48k_only_device_is_opened_at_48k_and_converted(monkeypatch):
    """The Digital Microphone case: it accepts 48 kHz and nothing else, so demanding
    16 kHz made a working device look unusable."""
    from localasr.capture import microphone

    opened = {}
    monkeypatch.setattr(
        microphone,
        "_sounddevice",
        lambda: _fake_sounddevice(48_000.0, accepts={48_000}, opened=opened),
    )

    level = microphone.probe_input(4, duration=0.5)

    assert opened["samplerate"] == 48_000, "device must be opened at a rate it supports"
    # The level is measured after conversion, so it reflects what the engine receives.
    assert level.rms == pytest.approx(0.01, rel=0.05)


def test_a_device_accepting_no_usual_rate_is_reported_not_crashed(monkeypatch):
    from localasr.capture import microphone

    monkeypatch.setattr(
        microphone,
        "_sounddevice",
        lambda: _fake_sounddevice(96_000.0, accepts=set(), opened={}),
    )

    with pytest.raises(microphone.CaptureError):
        microphone.probe_input(5, duration=0.1)


def _fake_pactl(monkeypatch, *, default_sink="bluez_output.00_42_79_F4_56_6D.1"):
    def fake(*args, **_kwargs):
        if args == ("get-default-sink",):
            return default_sink
        if args == ("list", "short", "sources"):
            return PACTL_SOURCES.strip()
        return ""

    monkeypatch.setattr(pulse, "_pactl", fake)


def test_monitors_are_read_from_pactl_not_from_portaudio(monkeypatch):
    """PortAudio here exposes only ALSA and OSS, so it never lists a PipeWire monitor
    however many pactl reports — searching it for one can only ever fail."""
    _fake_pactl(monkeypatch)
    monitors = pulse.list_monitors()
    assert len(monitors) == 2
    assert all(name.endswith(".monitor") for name in monitors)


def test_the_default_sinks_monitor_is_chosen(monkeypatch):
    """A laptop has several monitors; only the active sink carries the meeting."""
    _fake_pactl(monkeypatch)
    assert pulse.default_monitor() == "bluez_output.00_42_79_F4_56_6D.1.monitor"


def test_no_monitor_when_the_default_sink_has_none(monkeypatch):
    _fake_pactl(monkeypatch, default_sink="some-sink-without-a-monitor")
    assert pulse.default_monitor() is None


def test_missing_pactl_reports_no_monitor(monkeypatch):
    def boom(*_args, **_kwargs):
        raise pulse.PulseError("pactl not found")

    monkeypatch.setattr(pulse, "_pactl", boom)
    assert pulse.default_monitor() is None
    assert pulse.list_monitors() == []


def test_an_unreachable_server_is_not_reported_as_a_sink_without_a_monitor(monkeypatch):
    """Two causes, one None, and the wrong one was being named.

    `default_monitor()` returns None both when pactl cannot reach the audio server and
    when it reaches it and the default sink has no monitor. Doctor reported the second
    either way, which sends the user looking at their audio routing for a problem that
    is in the environment the process was started in — no XDG_RUNTIME_DIR, so no path to
    the server's socket at all. Observed under `env -u XDG_RUNTIME_DIR`.
    """
    from localasr.frontends.cli import doctor

    def unreachable(*_args, **_kwargs):
        raise pulse.PulseError("pactl info: Connection failure: Connection refused")

    monkeypatch.setattr(pulse, "_pactl", unreachable)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(doctor.shutil, "which", lambda name: f"/usr/bin/{name}")

    detail = _system_audio_detail(doctor)
    assert "Connection refused" in detail, "the server's own words are the finding"
    assert "no monitor for the default sink" not in detail
    assert "XDG_RUNTIME_DIR" in detail, "and the remedy that actually applies"


def test_a_reachable_server_with_no_monitor_still_says_so(monkeypatch):
    """The other branch has to survive: this one really is about audio routing, and
    the meeting can still be recorded from the microphone."""
    from localasr.frontends.cli import doctor

    _fake_pactl(monkeypatch, default_sink="some-sink-without-a-monitor")
    monkeypatch.setattr(doctor.shutil, "which", lambda name: f"/usr/bin/{name}")

    detail = _system_audio_detail(doctor)
    assert "no monitor for the default sink" in detail
    assert "XDG_RUNTIME_DIR" not in detail


def _system_audio_detail(doctor) -> str:
    check = next(c for c in doctor._capture_checks() if c.name.startswith("system audio"))
    assert not check.ok
    return check.detail


def test_parec_is_asked_for_the_pipeline_format(monkeypatch):
    """parec converts for us; asking for anything else would mean resampling here."""
    _fake_pactl(monkeypatch)
    monkeypatch.setattr(pulse.shutil, "which", lambda name: f"/usr/bin/{name}")
    seen = {}

    class FakeProcess:
        stdout = None
        stderr = None

        def poll(self):
            return 0

    def fake_popen(cmd, **_kwargs):
        seen["cmd"] = cmd
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    source = pulse.PulseMonitorSource()
    source.open()

    assert "--format=float32le" in seen["cmd"]
    assert "--rate=16000" in seen["cmd"]
    assert "--channels=1" in seen["cmd"]
    assert "--device=bluez_output.00_42_79_F4_56_6D.1.monitor" in seen["cmd"]


def test_opening_without_parec_raises(monkeypatch):
    monkeypatch.setattr(pulse, "available", lambda: False)
    with pytest.raises(pulse.PulseError, match="parec"):
        pulse.PulseMonitorSource().open()


def test_system_source_selection_falls_back_to_parec(monkeypatch):
    """PortAudio first, parec second — and on this desktop parec is the normal path."""
    from localasr.capture import microphone

    monkeypatch.setattr(microphone, "find_monitor", lambda: None)
    opened = {}

    class FakePulse:
        def __init__(self, device=None, source="system"):
            opened["device"] = device
            self.source = source

        def open(self):
            opened["opened"] = True

    monkeypatch.setattr(pulse, "PulseMonitorSource", FakePulse)
    source, warning = pulse.open_system_source(None)

    assert opened.get("opened")
    assert warning is None
    assert source.source == "system"


def test_system_source_reports_why_it_gave_up(monkeypatch):
    from localasr.capture import microphone

    monkeypatch.setattr(microphone, "find_monitor", lambda: None)

    class Failing:
        def __init__(self, **_kwargs):
            pass

        def open(self):
            raise pulse.PulseError("no monitor source for the default sink")

    monkeypatch.setattr(pulse, "PulseMonitorSource", Failing)
    source, warning = pulse.open_system_source(None)

    assert source is None
    assert "microphone only" in warning


# --- readiness ---------------------------------------------------------------


def test_ydotool_installed_but_unusable_is_not_reported_as_available(monkeypatch):
    """The package being present says nothing: it needs a running daemon and a
    writable /dev/uinput, and the latter needs a fresh login after a group change."""
    from localasr.platform import text_output

    monkeypatch.setattr(text_output.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(text_output.os, "access", lambda _path, _mode: False)

    ready, reason = text_output.ydotool_ready()
    assert not ready
    assert "uinput" in reason
    assert "ydotool" not in text_output.available_methods()


def test_the_uinput_remedy_is_a_group_change_not_a_system_service(monkeypatch):
    from localasr.platform import text_output

    monkeypatch.setattr(text_output.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(text_output.os, "access", lambda _path, _mode: False)

    remedy = dict((name, detail) for name, _ok, detail in text_output.diagnose())
    advice = remedy["ydotool (type at cursor)"]
    assert "usermod -aG input" in advice
    assert "log out" in advice
    assert "sudo systemctl enable" not in advice


def test_a_missing_daemon_is_reported_as_a_user_service(monkeypatch, tmp_path):
    from localasr.platform import text_output

    monkeypatch.setattr(text_output.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(text_output.os, "access", lambda _path, _mode: True)
    monkeypatch.setenv("YDOTOOL_SOCKET", str(tmp_path / "absent.sock"))

    remedy = dict((name, detail) for name, _ok, detail in text_output.diagnose())
    assert "systemctl --user enable --now ydotool.service" in remedy["ydotool (type at cursor)"]


def test_socket_resolution_matches_what_ydotool_itself_would_do(monkeypatch, tmp_path):
    """ydotool falls back to /tmp when XDG_RUNTIME_DIR is gone; the check must too.

    A hardcoded /run/user/<uid> path made ydotool_ready() approve a socket the
    command would never open, so dictation failed into the clipboard whenever the
    process had no XDG_RUNTIME_DIR (systemd system unit, cron, tty, ssh).
    """
    from localasr.platform import text_output

    monkeypatch.delenv("YDOTOOL_SOCKET", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    assert text_output.ydotool_socket() == "/tmp/.ydotool_socket"

    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert text_output.ydotool_socket() == f"{tmp_path}/.ydotool_socket"

    monkeypatch.setenv("YDOTOOL_SOCKET", "/explicit.sock")
    assert text_output.ydotool_socket() == "/explicit.sock"


def test_typing_is_pinned_to_the_socket_the_check_approved(monkeypatch, tmp_path):
    """The child must not re-resolve the socket and reach a different one."""
    from localasr.platform import text_output

    monkeypatch.delenv("YDOTOOL_SOCKET", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    (tmp_path / ".ydotool_socket").touch()

    seen: dict[str, str] = {}

    def fake_run(cmd, text=None, timeout=10.0, env=None):
        seen.update(env or {})
        return True

    monkeypatch.setattr(text_output, "_run", fake_run)
    assert text_output._type_with_ydotool("hi")
    assert seen["YDOTOOL_SOCKET"] == text_output.ydotool_socket()


def test_missing_clipboard_is_reported_because_it_is_the_last_resort(monkeypatch):
    from localasr.platform import text_output

    monkeypatch.setattr(text_output.shutil, "which", lambda _name: None)
    remedy = dict((name, detail) for name, _ok, detail in text_output.diagnose())
    assert "wl-clipboard" in remedy["clipboard fallback"]


def test_doctor_separates_what_each_application_needs():
    checks = doctor.run_checks()
    needed_by_subtitle = {c.name for c in checks if "subtitle" in c.required_by}
    needed_by_dictate = {c.name for c in checks if "dictate" in c.required_by}

    assert "ffmpeg" in needed_by_subtitle
    assert "ffmpeg" not in needed_by_dictate
    assert any("ydotool" in name for name in needed_by_dictate)
    assert not any("ydotool" in name for name in needed_by_subtitle)


def test_text_delivery_never_blocks_subtitles():
    """A machine with no ydotool must still be able to make subtitles."""
    checks = doctor.run_checks()
    assert not any("ydotool" in c.name for c in doctor.blocking_for("subtitle", checks))


# --- desktop entries ---------------------------------------------------------


def test_three_launchers_are_written(tmp_path):
    written = entries.install(tmp_path / "applications", tmp_path / "icons")
    desktop_files = [p for p in written if p.suffix == ".desktop"]
    assert len(desktop_files) == 3


def test_every_launcher_targets_the_same_single_instance_host(tmp_path):
    """Three separate programs would mean three llama-servers."""
    entries.install(tmp_path / "applications", tmp_path / "icons")
    execs = [
        line
        for path in sorted((tmp_path / "applications").glob("*.desktop"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("Exec=")
    ]
    assert len(execs) == 3
    assert all("localasr-desktop" in line for line in execs)
    assert {line.rsplit(" ", 1)[-1] for line in execs} == {"subtitle", "meeting", "dictation"}


def test_launchers_are_removable(tmp_path):
    entries.install(tmp_path / "applications", tmp_path / "icons")
    removed = entries.uninstall(tmp_path / "applications", tmp_path / "icons")
    assert len(removed) == 4
    assert not list((tmp_path / "applications").glob("*.desktop"))
