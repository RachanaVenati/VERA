"""TIRA CLI wrapper for Docker logging and upload timeout compatibility.

Docker SDK's synchronous ContainerCollection.run returns None with that driver;
TIRA 0.0.205 expects bytes in its tracker probe. Use json-file per container,
without changing the user's daemon settings or submission image.
"""
import os

import docker
from docker.models.containers import ContainerCollection
from tira.tira_cli import main

original_run = ContainerCollection.run
original_from_env = docker.from_env


def docker_with_timeout(*args, **kwargs):
    # Docker SDK defaults to 60 seconds without a response. Large registry
    # pushes can legitimately go longer without emitting a progress update.
    kwargs.setdefault("timeout", int(os.environ.get("VERA_DOCKER_TIMEOUT_SECONDS", "600")))
    return original_from_env(*args, **kwargs)


def run_with_logs(self, *args, **kwargs):
    kwargs.setdefault("log_config", {"type": "json-file", "config": {}})
    return original_run(self, *args, **kwargs)


if __name__ == "__main__":
    docker.from_env = docker_with_timeout
    ContainerCollection.run = run_with_logs
    raise SystemExit(main())
