"""`fluke-get` command line interface."""
import os

import requests
import rich
import typer
import yaml

app = typer.Typer()


@app.command()
def list():
    """
    List all available configuration files.
    """

    url = "https://api.github.com/repos/makgyver/fluke/contents/configs"
    response = requests.get(url, timeout=5)
    response.raise_for_status()

    configs = [file["name"].removesuffix('.yaml')
               for file in response.json() if file["name"].endswith(".yaml")]

    rich.print("[yellow bold]Available config files:[/]")
    for config in configs:
        rich.print(config)


@app.command()
def config(name: str, outdir: str = typer.Option("config", help="Output directory")):
    """
    Get a configuration file by name.
    """

    url = f"https://raw.githubusercontent.com/makgyver/fluke/main/configs/{name}.yaml"

    rich.print(f"Getting config file from {url} ...")
    response = requests.get(url, timeout=5)

    if response.status_code != 200:
        rich.print(f"[red][Error]:[/] [yellow]config file {name} not found.[/]")
        return

    config = yaml.safe_load(response.text)

    if not config:
        rich.print("Config file is empty.")
        return

    # if outdir does not exist, create it
    if not os.path.exists(outdir):
        rich.print(f"Creating output directory {outdir} ...")
        os.makedirs(outdir)

    if os.path.exists(f"{outdir}/{name}.yaml"):
        rich.print(
            "[red][Error]:[/] [yellow]refusing to overwrite existing config" +
            f"file {outdir}/{name}.yaml. Please rename it or delete it.[/]")
        return

    rich.print(f"Saving config file to {outdir}/{name}.yaml ...")
    with open(f"{outdir}/{name}.yaml", "w", encoding="utf8") as f:
        yaml.dump(config, f)


def main():
    app()


if __name__ == "__main__":
    app()
