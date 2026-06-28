"""
CLI entry point for the Telemetry Emulator.

Commands:
    telem-emu generate [--scenario] [--persona] [--cycles] [--sessions]
    telem-emu capture register <element> <screenshot> [...]
    telem-emu capture batch <dir>
    telem-emu capture list
    telem-emu config show
    telem-emu config init [--output]

Usage:
    python -m tools.cli generate --scenario chaos_altar_dragon_bones --cycles 5
    python -m tools.cli capture list
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np

from core.config import AppConfig, PersonaType, DEFAULT_PERSONAS
from core.scheduler import SessionScheduler, SessionMemory, SessionRecord


@click.group()
@click.option("--config", "-c", "config_path", type=click.Path(exists=True),
              help="Path to YAML config file.")
@click.option("--data-dir", "-d", type=click.Path(), default="data",
              help="Path to data directory.")
@click.option("--output-dir", "-o", type=click.Path(), default="output",
              help="Path to output directory.")
@click.pass_context
def main(ctx, config_path, data_dir, output_dir):
    """Synthetic Human Telemetry Emulator — generate training data."""
    ctx.ensure_object(dict)
    if config_path:
        ctx.obj["config"] = AppConfig.from_yaml(config_path)
    else:
        ctx.obj["config"] = AppConfig()
    ctx.obj["config"].data_dir = data_dir
    ctx.obj["config"].output_dir = output_dir
    ctx.obj["data_dir"] = Path(data_dir)
    ctx.obj["output_dir"] = Path(output_dir)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

@main.command()
@click.option("--scenario", "-s", default="chaos_altar_dragon_bones",
              help="Scenario to run.")
@click.option("--persona", "-p", default="fast_accurate",
              type=click.Choice(["fast_accurate", "slow_methodical", "distracted_error_prone"]),
              help="Persona to simulate.")
@click.option("--cycles", "-n", default=10, type=int, help="Number of cycles.")
@click.option("--sessions", default=1, type=int, help="Number of sessions.")
@click.option("--seed", default=None, type=int, help="RNG seed.")
@click.option("--synthetic/--live", default=True, help="Synthetic vs live screen mode.")
@click.pass_context
def generate(ctx, scenario, persona, cycles, sessions, seed, synthetic):
    """Generate synthetic telemetry data."""
    config: AppConfig = ctx.obj["config"]
    output_dir = ctx.obj["output_dir"]

    persona_type = PersonaType(persona)
    rng = np.random.default_rng(seed)

    for i in range(sessions):
        session_seed = rng.integers(0, 2**31) if seed is None else seed + i

        click.echo(f"Session {i+1}/{sessions}: persona={persona}, cycles={cycles}")

        if scenario == "chaos_altar_dragon_bones":
            from scenarios.chaos_altar_dragon_bones import create_chaos_altar_session
            summary = create_chaos_altar_session(
                config=config,
                persona_type=persona_type,
                cycles=cycles,
                output_dir=str(output_dir),
                synthetic=synthetic,
                seed=session_seed,
            )
        else:
            click.echo(f"Unknown scenario: {scenario}", err=True)
            continue

        click.echo(f"  → {json.dumps(summary, indent=2)}")


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------

@main.group()
def capture():
    """Template capture and management tools."""
    pass


@capture.command("register")
@click.argument("element")
@click.argument("screenshot", type=click.Path(exists=True))
@click.option("--state", default="default", help="UI state.")
@click.option("--yaw", default=0, type=int, help="Camera yaw (degrees).")
@click.option("--pitch", default="default", help="Camera pitch.")
@click.option("--zoom", default="mid", help="Zoom level.")
@click.option("--bbox", default="0,0,100,100", help="Bounding box: x,y,w,h.")
@click.pass_context
def capture_register(ctx, element, screenshot, state, yaw, pitch, zoom, bbox):
    """Register a new template variant from a screenshot."""
    from tools.template_capture import TemplateCaptureTool

    parts = [int(x.strip()) for x in bbox.split(",")]
    if len(parts) != 4:
        click.echo("Error: --bbox must be 'x,y,w,h'", err=True)
        return

    tool = TemplateCaptureTool(ctx.obj["data_dir"] / "templates")
    result = tool.register_from_screenshot(
        element_name=element,
        screenshot_path=screenshot,
        bbox=tuple(parts),
        state=state,
        yaw=yaw,
        pitch=pitch,
        zoom=zoom,
    )
    click.echo(f"Registered: {result}")


@capture.command("batch")
@click.argument("directory", type=click.Path(exists=True))
@click.option("--auto-detect/--manual", default=False, help="Auto-detect regions.")
@click.pass_context
def capture_batch(ctx, directory, auto_detect):
    """Process all screenshots in a directory."""
    from tools.template_capture import TemplateCaptureTool

    tool = TemplateCaptureTool(ctx.obj["data_dir"] / "templates")
    tool.batch_register(directory, auto_detect=auto_detect)
    click.echo("Batch complete.")


@capture.command("list")
@click.pass_context
def capture_list(ctx):
    """List all registered templates."""
    from tools.template_capture import TemplateCaptureTool

    tool = TemplateCaptureTool(ctx.obj["data_dir"] / "templates")
    elements = tool.list_elements()

    if not elements:
        click.echo("No templates registered.")
        return

    click.echo(f"{'Element':<30} {'Variants':<10} {'States'}")
    click.echo("-" * 60)
    for el in elements:
        click.echo(f"{el['name']:<30} {el['variant_count']:<10} {', '.join(el['states'])}")


@capture.command("validate")
@click.argument("element")
@click.pass_context
def capture_validate(ctx, element):
    """Validate a template element."""
    from tools.template_capture import TemplateCaptureTool

    tool = TemplateCaptureTool(ctx.obj["data_dir"] / "templates")
    warnings = tool.validate_element(element)
    if warnings:
        for w in warnings:
            click.echo(f"  {w}")
    else:
        click.echo(f"'{element}' is valid.")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@main.group()
def config():
    """Configuration management."""
    pass


@config.command("show")
@click.pass_context
def config_show(ctx):
    """Show current configuration."""
    cfg: AppConfig = ctx.obj["config"]
    click.echo(json.dumps(cfg.to_dict(), indent=2, default=str))


@config.command("init")
@click.option("--output", "-o", default="config.yaml", type=click.Path(),
              help="Output path for default config.")
@click.pass_context
def config_init(ctx, output):
    """Write default configuration to a YAML file."""
    cfg: AppConfig = ctx.obj["config"]
    cfg.to_yaml(output)
    click.echo(f"Config written to: {output}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
