import sys

import typer

from . import transforms
from .cli import app as cli_app
from .cli.synapse import app as synapse_app
from .serving.cli import app as serve_app
from .web.cli import app as webcli_app
from .web.ui import app as webui_app

app = typer.Typer(help="Brain surgery command suite.")
app.add_typer(cli_app, name="cli")
app.add_typer(synapse_app, name="synapse")
app.add_typer(serve_app, name="serve")
app.add_typer(webcli_app, name="webcli")
app.add_typer(webui_app, name="webui")


def _normalize_cli_args(raw_args: list[str]) -> list[str]:
    option_tokens: list[str] = []
    positional_tokens: list[str] = []
    index = 0
    while index < len(raw_args):
        token = raw_args[index]
        if token == "--":
            positional_tokens.extend(raw_args[index:])
            break
        if token.startswith("--"):
            option_tokens.append(token)
            if "=" not in token and index + 1 < len(raw_args):
                next_token = raw_args[index + 1]
                if not next_token.startswith("-"):
                    option_tokens.append(next_token)
                    index += 2
                    continue
            index += 1
            continue
        if token.startswith("-") and token != "-":
            option_tokens.append(token)
            index += 1
            continue

        positional_tokens.append(token)
        index += 1

    return [*option_tokens, *positional_tokens]


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    top_level = {
        "cli",
        "synapse",
        "serve",
        "webcli",
        "webui",
        "-h",
        "--help",
        "--install-completion",
        "--show-completion",
    }
    if args and args[0] in top_level:
        if args[0] == "cli":
            app(args=["cli", *_normalize_cli_args(args[1:])], prog_name="brainsurgery")
            return
        app(args=args, prog_name="brainsurgery")
        return
    app(args=["cli", *_normalize_cli_args(args)], prog_name="brainsurgery")


__all__ = ["app", "main", "cli_app", "synapse_app", "serve_app", "webcli_app", "webui_app", "transforms"]
