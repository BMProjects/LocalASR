"""localasr command line.

Three application entry points — `subtitle`, `dictate`, `meeting` — over one shared
context, plus model management. Each is a separate command because their inputs,
lifetimes and outputs genuinely differ; forcing them through one interface is what the
first version got wrong.
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path
from typing import Annotated

import typer

from localasr.apps.coordinator import ActivityConflict
from localasr.apps.dictation import DictationController, DictationOptions
from localasr.apps.meeting import (
    MeetingController,
    MeetingOptions,
    default_journal_path,
    load_journal,
)
from localasr.apps.subtitle import SubtitleController, SubtitleOptions
from localasr.context import AppContext
from localasr.core.engine import lock
from localasr.formats.subtitles import WRITERS, render
from localasr.frontends.cli.reporting import live_reporter, offline_reporter
from localasr.registry import manager

app = typer.Typer(add_completion=False, help="Local-first speech recognition.")
models_app = typer.Typer(help="Inspect, download and verify models.")
audio_app = typer.Typer(help="Inspect audio devices.")
app.add_typer(models_app, name="models")
app.add_typer(audio_app, name="audio")


def _context() -> AppContext:
    return AppContext()


@app.command()
def subtitle(
    media: Annotated[list[Path], typer.Argument(help="Files or directories.")],
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Output file.")] = None,
    outdir: Annotated[Path | None, typer.Option("--outdir", help="Output directory.")] = None,
    fmt: Annotated[str, typer.Option("-f", "--format", help=f"{sorted(WRITERS)}")] = "srt",
    language: Annotated[str | None, typer.Option("-l", "--language")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    no_resume: Annotated[bool, typer.Option("--no-resume")] = False,
    quiet: Annotated[bool, typer.Option("-q", "--quiet")] = False,
) -> None:
    """Generate subtitles for one or more media files."""
    if output is not None and len(media) == 1 and media[0].is_file():
        fmt = output.suffix.lstrip(".").lower() or fmt
        outdir = output.parent

    context = _context()
    controller = SubtitleController(
        context,
        SubtitleOptions(
            fmt=fmt,
            language=language or context.settings.language,
            output_dir=outdir,
            overwrite=overwrite,
            resume=not no_resume,
        ),
    )

    signal.signal(signal.SIGINT, lambda *_a: controller.cancel())
    try:
        results = controller.run(media, listener=offline_reporter(quiet))
    finally:
        context.shutdown()

    failed = 0
    for result in results:
        if result.skipped:
            typer.echo(f"skip   {result.media.name} (output exists; --overwrite to replace)")
        elif result.error:
            failed += 1
            typer.secho(f"fail   {result.media.name}: {result.error}", fg="red")
        else:
            if output is not None and len(results) == 1:
                result.output.rename(output)
                typer.echo(f"wrote  {output}")
            else:
                typer.echo(f"wrote  {result.output}")
    if failed:
        raise typer.Exit(1)


@app.command()
def transcribe(
    media: Annotated[Path, typer.Argument()],
    output: Annotated[Path | None, typer.Option("-o", "--output")] = None,
    fmt: Annotated[str | None, typer.Option("-f", "--format")] = None,
    language: Annotated[str | None, typer.Option("-l", "--language")] = None,
    quiet: Annotated[bool, typer.Option("-q", "--quiet")] = False,
) -> None:
    """Deprecated alias for `subtitle`, kept so existing scripts keep working."""
    if output is None:
        context = _context()
        controller = SubtitleController(
            context, SubtitleOptions(fmt=fmt or "txt", language=language, overwrite=True)
        )
        try:
            results = controller.run([media], listener=offline_reporter(quiet))
        finally:
            context.shutdown()
        result = results[0]
        if result.transcript is not None:
            typer.echo(render(result.transcript, fmt or "txt"), nl=False)
            result.output.unlink(missing_ok=True)
        return
    subtitle(
        media=[media],
        output=output,
        outdir=None,
        fmt=fmt or output.suffix.lstrip(".") or "srt",
        language=language,
        overwrite=True,
        no_resume=False,
        quiet=quiet,
    )


@app.command()
def dictate(
    seconds: Annotated[
        float | None, typer.Option("--seconds", help="Record for a fixed time, then stop.")
    ] = None,
    language: Annotated[str | None, typer.Option("-l", "--language")] = None,
    device: Annotated[str | None, typer.Option("-d", "--device", help="Input device.")] = None,
    no_deliver: Annotated[
        bool, typer.Option("--no-deliver", help="Print text instead of typing it.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Start even if another workload is running.")
    ] = False,
    with_system: Annotated[
        bool,
        typer.Option("--with-system", help="Also transcribe what the computer is playing."),
    ] = False,
    check: Annotated[bool, typer.Option("--check", help="Report text-delivery setup.")] = False,
) -> None:
    """Push-to-talk dictation: records until Enter (or --seconds), then types the text."""
    if check:
        from localasr.frontends.cli.doctor import hotkey_hint, run_checks

        for item in run_checks():
            if "dictate" not in item.required_by:
                continue
            mark = typer.style("ok  ", fg="green") if item.ok else typer.style("FAIL", fg="red")
            typer.echo(f"{mark} {item.name}")
            for line in item.detail.splitlines():
                typer.echo(f"       {line}")
        typer.echo("")
        typer.echo(hotkey_hint())
        return

    context = _context()
    controller = DictationController(
        context,
        DictationOptions(
            language=language,
            device=device,
            deliver=not no_deliver,
            capture_system=with_system,
        ),
        listener=lambda event: None,
    )

    try:
        controller.start(force=force)
    except ActivityConflict as exc:
        typer.secho(f"{exc}; rerun with --force to interrupt it", fg="yellow", err=True)
        raise typer.Exit(2) from exc

    typer.secho("● recording — press Enter to stop", fg="red", err=True)
    try:
        if seconds is not None:
            time.sleep(seconds)
        else:
            sys.stdin.readline()
    except KeyboardInterrupt:
        controller.cancel()
        context.shutdown()
        raise typer.Exit(130) from None

    try:
        text = controller.finish()
    finally:
        context.shutdown()

    if not text:
        typer.secho("nothing recognised", fg="yellow", err=True)
        raise typer.Exit(1)
    if no_deliver:
        typer.echo(text)
    else:
        typer.secho(f"→ {text}", err=True)


@app.command()
def meeting(
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Journal path.")] = None,
    minutes: Annotated[
        float | None, typer.Option("--minutes", help="Stop after N minutes.")
    ] = None,
    language: Annotated[str | None, typer.Option("-l", "--language")] = None,
    mic: Annotated[str | None, typer.Option("--mic", help="Microphone device.")] = None,
    system: Annotated[str | None, typer.Option("--system", help="System-audio device.")] = None,
    no_system: Annotated[
        bool, typer.Option("--no-system", help="Record the microphone only.")
    ] = False,
    force_system: Annotated[
        bool,
        typer.Option(
            "--force-system",
            help="Record system audio even if the microphone can already hear it.",
        ),
    ] = False,
    export: Annotated[
        str, typer.Option("--export", help=f"Export format on stop: {sorted(WRITERS)}")
    ] = "md",
    quiet: Annotated[bool, typer.Option("-q", "--quiet")] = False,
) -> None:
    """Record a meeting: microphone tagged 我, system audio tagged 对方."""
    context = _context()
    journal = output or default_journal_path()
    controller = MeetingController(
        context,
        MeetingOptions(
            language=language,
            mic_device=mic,
            system_device=system,
            capture_system=not no_system,
            check_overlap=not force_system,
        ),
        listener=live_reporter(quiet),
    )

    session = controller.start(journal)
    for warning in session.warnings:
        typer.secho(f"warning: {warning}", fg="yellow", err=True)
    typer.secho(
        "recording — 我 = microphone, 对方 = system audio (source labels, not speaker "
        "separation). Press Enter to stop.",
        fg="green",
        err=True,
    )

    try:
        if minutes is not None:
            time.sleep(minutes * 60)
        else:
            sys.stdin.readline()
    except KeyboardInterrupt:
        pass

    try:
        finished = controller.stop()
    finally:
        context.shutdown()

    if finished is None or not finished.segments:
        typer.secho(f"no speech recorded; journal at {journal}", fg="yellow", err=True)
        return

    target = journal.with_suffix(f".{export}")
    target.write_text(render(finished.transcript(), export), encoding="utf-8")
    typer.echo(f"journal  {journal}")
    typer.echo(f"export   {target}")


@app.command("export")
def export_meeting(
    journal: Annotated[Path, typer.Argument(help="Meeting journal (.jsonl).")],
    fmt: Annotated[str, typer.Option("-f", "--format", help=f"{sorted(WRITERS)}")] = "md",
    output: Annotated[Path | None, typer.Option("-o", "--output")] = None,
) -> None:
    """Re-export a meeting journal, including one left behind by a crash."""
    session = load_journal(journal)
    target = output or journal.with_suffix(f".{fmt}")
    target.write_text(render(session.transcript(), fmt), encoding="utf-8")
    typer.echo(f"{len(session.segments)} segment(s) -> {target}")


@app.command()
def desktop(
    which: Annotated[str, typer.Argument(help="subtitle | meeting | dictation")] = "subtitle",
    quit_running: Annotated[
        bool, typer.Option("--quit", help="Ask a running desktop host to exit.")
    ] = False,
) -> None:
    """Launch the desktop host: one process, three applications.

    A second invocation raises the requested window in the already-running host rather
    than starting a rival process with its own engine.
    """
    from localasr.frontends.desktop.app import main as desktop_main
    from localasr.frontends.desktop.single_instance import QUIT

    raise typer.Exit(desktop_main([QUIT if quit_running else which]))


@app.command()
def doctor(
    app_name: Annotated[
        str | None, typer.Argument(help="Only check one of: subtitle, dictate, meeting.")
    ] = None,
) -> None:
    """Check what each application needs, and print the command that fixes what is missing."""
    from localasr.frontends.cli.doctor import blocking_for, hotkey_hint, run_checks

    checks = run_checks()
    wanted = {app_name} if app_name else {"subtitle", "dictate", "meeting"}

    for check in checks:
        if not (set(check.required_by) & wanted):
            continue
        mark = typer.style("ok  ", fg="green") if check.ok else typer.style("FAIL", fg="red")
        used_by = ",".join(check.required_by)
        typer.echo(f"{mark} {check.name:<28} [{used_by}]")
        if not check.ok or "\n" in check.detail:
            for line in check.detail.splitlines():
                typer.echo(f"       {line}")
        elif check.detail not in {"ready"}:
            typer.echo(f"       {check.detail}")

    typer.echo("")
    for name in ("subtitle", "dictate", "meeting"):
        if name not in wanted:
            continue
        blocking = blocking_for(name, checks)
        if blocking:
            missing = ", ".join(c.name for c in blocking)
            typer.secho(f"localasr {name}: 未就绪 — 缺 {missing}", fg="red")
        else:
            typer.secho(f"localasr {name}: 就绪", fg="green")

    if "dictate" in wanted:
        typer.echo("")
        typer.echo(hotkey_hint())


@app.command("install-desktop")
def install_desktop(
    remove: Annotated[bool, typer.Option("--remove", help="Delete the launchers.")] = False,
) -> None:
    """Install application-menu launchers for the three applications."""
    from localasr.frontends.desktop import entries

    if remove:
        for path in entries.uninstall():
            typer.echo(f"removed {path}")
        return

    written = entries.install()
    for path in written:
        typer.echo(f"wrote {path}")
    typer.echo("")
    typer.echo("在应用菜单中搜索 “LocalASR” 即可看到三个入口。")


@models_app.command("list")
def models_list() -> None:
    """List catalog models, their pinned revision and local state."""
    for spec in manager.list_models():
        state = "downloaded" if manager.is_downloaded(spec) else "not downloaded"
        marker = "*" if spec.default else " "
        # No measured column: it was fed by a `localasr bench` that does not exist, so
        # every row read "unmeasured" forever. The one number that is measured — the
        # memory overhead factor — lives with the guard that uses it.
        typer.echo(
            f"{marker} {spec.model_id:<22} {spec.kind.value:<4} "
            f"{spec.revision[:8]}  {state}"
        )


@models_app.command("pull")
def models_pull(model_id: Annotated[str | None, typer.Argument()] = None) -> None:
    """Download a model's weights, verifying each file against the catalog."""
    spec = manager.get_model(model_id)
    typer.echo(f"pulling {spec.model_id} from {spec.repo}@{spec.revision[:8]} ...", err=True)
    model_path, mmproj_path = manager.pull_model(spec)
    typer.echo(f"  {model_path}")
    if mmproj_path:
        typer.echo(f"  {mmproj_path}")


@models_app.command("verify")
def models_verify(model_id: Annotated[str | None, typer.Argument()] = None) -> None:
    """Re-hash a downloaded model and compare against the pinned checksums."""
    spec = manager.get_model(model_id)
    try:
        manager.verify_model(spec)
    except manager.IntegrityError as exc:
        typer.secho(f"FAILED  {spec.model_id}: {exc}", fg="red", err=True)
        raise typer.Exit(1) from exc
    typer.secho(f"ok  {spec.model_id} matches {spec.revision[:8]}", fg="green")


@models_app.command("import")
def models_import(
    model_file: Annotated[Path, typer.Argument(help="模型权重（.gguf）。")],
    mmproj: Annotated[
        Path | None, typer.Option("--mmproj", help="配套的 mmproj 文件；音频模型必需。")
    ] = None,
    name: Annotated[str | None, typer.Option("--name", help="列表中显示的名字。")] = None,
    kind: Annotated[
        str, typer.Option("--kind", help="模型角色：asr（默认）或 llm（整理）。")
    ] = "asr",
    link: Annotated[
        bool,
        typer.Option(
            "--link",
            help="链接而不是复制。适用于文件由别的工具管理时，避免在同一块盘上多存一份。",
        ),
    ] = False,
) -> None:
    """Import a GGUF you already have on disk."""
    from localasr.registry import imported

    try:
        role = manager.ModelKind(kind)
    except ValueError as exc:
        typer.secho(f"未知的 --kind：{kind}（可选 asr / llm）", fg="red", err=True)
        raise typer.Exit(1) from exc

    owner = imported.in_store(model_file)
    if owner and not link:
        # Copying out of a store nobody is going to delete duplicates several GB on the
        # same disk for nothing. Said, not decided: --link is still the user's to pass.
        typer.secho(
            f"提示：这个文件在 {owner} 的模型库里，加 --link 可避免重复占用磁盘。",
            fg="yellow",
            err=True,
        )

    request = imported.ImportRequest(
        model_path=model_file, mmproj_path=mmproj, name=name, kind=role, link=link
    )
    try:
        def report(item: str, done: int, total: int) -> None:
            typer.echo(f"  [{done}/{total}] {item}", err=True)

        spec = imported.import_model(request, progress=report)
    except manager.RegistryError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(1) from exc
    how = "链接" if link else "复制"
    typer.secho(
        f"已导入 {spec.model_id}（{how}，角色 {spec.kind.value}，{manager.model_dir(spec)}）",
        fg="green",
    )


@models_app.command("remove")
def models_remove(
    model_id: Annotated[str, typer.Argument(help="要卸载的模型 id。")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="不再询问。")] = False,
) -> None:
    """Delete a model's downloaded weights."""
    spec = manager.get_model(model_id)
    if not manager.is_downloaded(spec):
        typer.echo(f"{model_id} 未下载，无需卸载")
        return
    size = sum(file.size for file in spec.files)
    if not yes:
        typer.confirm(f"删除 {model_id} 的权重（约 {size / 1e9:.2f} GB）？", abort=True)
    freed = manager.remove_model(spec)
    typer.secho(f"已卸载 {model_id}，回收 {freed / 1e9:.2f} GB", fg="green")


@models_app.command("status")
def models_status() -> None:
    """Show whether an engine is already running, and which model it holds."""
    record = lock.read()
    if record is None:
        typer.echo("no engine running")
        return
    typer.echo(f"engine pid {record.pid} at {record.base_url}")
    typer.echo(f"  model {record.model_id} @ {record.revision[:8]}")


@audio_app.command("list")
def audio_list() -> None:
    """List input devices. Monitor devices are how system audio is captured."""
    from localasr.capture.microphone import CaptureError, list_devices
    from localasr.capture.naming import device_labels

    try:
        devices = list_devices()
    except CaptureError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(1) from exc
    for device, label in zip(devices, device_labels(devices), strict=True):
        typer.echo(f"[{device.index}] {label}")


if __name__ == "__main__":
    app()
