import typer
from replicate_runner.logger_config import create_logger
from replicate_runner.commands import replicate_cmds, hf_cmds
from rich.console import Console

app = typer.Typer()
app.add_typer(replicate_cmds.app, name="replicate")
app.add_typer(hf_cmds.app, name="hf")

console = Console()

@app.callback()
def main_callback():
    """
    Main callback to initialize the logger, any global stuff, etc.
    This runs before any subcommand.
    """
    create_logger()
    console.print("[bold blue]Welcome to Replicate Runner![/bold blue]")

def run():
    """
    The run function that calls typer to handle CLI arguments.
    """
    app()

if __name__ == "__main__":
    run()
