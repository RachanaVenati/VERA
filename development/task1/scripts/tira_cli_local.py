"""TIRA CLI wrapper for Docker Desktop's `local` logging-driver compatibility.

Docker SDK's synchronous ContainerCollection.run returns None with that driver;
TIRA 0.0.205 expects bytes in its tracker probe. Use json-file per container,
without changing the user's daemon settings or submission image.
"""
from docker.models.containers import ContainerCollection
from tira.tira_cli import main

original_run = ContainerCollection.run


def run_with_logs(self, *args, **kwargs):
    kwargs.setdefault("log_config", {"type": "json-file", "config": {}})
    return original_run(self, *args, **kwargs)


if __name__ == "__main__":
    ContainerCollection.run = run_with_logs
    raise SystemExit(main())
