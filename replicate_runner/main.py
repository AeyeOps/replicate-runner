import typer
from replicate_runner.logger_config import create_logger
from replicate_runner.commands import (
    replicate_cmds,
    hf_cmds,
    profile_cmds,
    prompt_cmds,
    explore_cmds,
)
from rich.console import Console

app = typer.Typer(invoke_without_command=True)
app.add_typer(replicate_cmds.app, name="replicate")
app.add_typer(hf_cmds.app, name="hf")
app.add_typer(profile_cmds.app, name="profile")
app.add_typer(prompt_cmds.app, name="prompt")
app.add_typer(explore_cmds.app, name="explore")

console = Console()

@app.callback()
def main_callback(ctx: typer.Context):
    """
    Main callback to initialize the logger, any global stuff, etc.
    This runs before any subcommand.
    """
    create_logger()
    console.print("[bold blue]Welcome to Replicate Runner![/bold blue]")
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())

def run():
    """
    The run function that calls typer to handle CLI arguments.
    """
    app()

if __name__ == "__main__":
    run()
