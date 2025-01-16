import typer

app = typer.Typer()


@app.command()
def hello(name: str):
    typer.echo(f"Hello {build_string(name, 1)}!")


def build_string(name: str, repeat: int) -> str:
    return name * repeat


def main():
    app()


if __name__ == "__main__":
    main()
