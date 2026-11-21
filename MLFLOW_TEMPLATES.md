
# MLflow server templates (TEMPORARY holding file)

> **NOTE (2026-11-21):** this content was moved out of `README.md`, which now
> holds the project documentation. The MLflow tracking server is being
> migrated to a new version installed elsewhere; the templates below (run
> creation, probe-timeout patch, run termination) describe how to work with
> that server. This file is temporary and may be reorganized or removed once
> the new server setup is permanently documented.

---

# Reparameterized FWI
Simple Encoder-Decoder Architecture For Velocity Foundation Models



## MLFLOW Amazon (examples from Alan)

### Template 1
import argparse
import os

import mlflow
from mlflow.environment_variables import MLFLOW_RUN_ID
from mlflow.tracking import MlflowClient

from interpam.mlflow_probe_patch import apply as patch_probe_timeout


def getargs():
    parser = argparse.ArgumentParser(
        description="create an mlflow run and export its id for subsequent steps"
    )

    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        required=False,
        help="mlflow tracking server uri (defaults to MLFLOW_TRACKING_URI env var)",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="Default",
        required=False,
        help="mlflow experiment name",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        required=False,
        help="mlflow run name",
    )

    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        required=False,
        help="if set, also append MLFLOW_RUN_ID=<run_id> to this file (e.g. for CI job steps)",
    )

    parser.add_argument(
        "--finish",
        action="store_true",
        required=False,
        help="terminate the run as FINISHED after creation (default: leave it RUNNING so "
        "subsequent steps sharing MLFLOW_RUN_ID can keep logging; use finish_mlflow_run "
        "in the last step to terminate it)",
    )

    return parser.parse_args()


def main():
    args = getargs()

    patch_probe_timeout()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id

    if not args.finish:
        MlflowClient().set_terminated(run_id, status="RUNNING")

    client = MlflowClient()
    run = client.get_run(run_id)
    print(f"MLFLOW_RUN_ID={run_id}")
    print(f"run status: {run.info.status}")

    os.environ[MLFLOW_RUN_ID.name] = run_id

    if args.env_file:
        with open(args.env_file, "a") as f:
            f.write(f"export {MLFLOW_RUN_ID.name}={run_id}\n")


if __name__ == "__main__":
    main()
 

### Template 2
"""Shared patch: raise MLflow's hardcoded 3s workspace-probe timeout.

MLflow's client-side workspace probe (_probe_workspace_support ->
fetch_server_info, mlflow/store/workspace_rest_store_mixin.py) calls
/api/3.0/mlflow/server-info with a HARDCODED timeout=3
(mlflow/utils/server_info.py) that ignores MLFLOW_HTTP_REQUEST_TIMEOUT.
Through the shared LB the call takes 2.2-3.8s, so any client that sets
MLFLOW_WORKSPACE intermittently fails with "Failed to query
/api/3.0/mlflow/server-info" on that race.

The patch must bind http_request in the mlflow.utils.server_info namespace
(that module does `from mlflow.utils.rest_utils import http_request` at
import time, so patching rest_utils alone has no effect).

Config (read at apply() time): MLFLOW_PROBE_TIMEOUT (default 30),
MLFLOW_PROBE_MAX_RETRIES (default 3). CLI: python -m interpam.mlflow_probe_patch
(--install/--remove/--check) — the .pth install makes the patch active in
EVERY python process of the environment (devcontainer, Slurm venv, a future
K8s client-job image); it is lazy, so processes that never import mlflow
pay ~nothing.

Usage from the CLIs in this package (create_mlflow_run / finish_mlflow_run):

    from interpam.mlflow_probe_patch import apply
    apply()

Same patch logic as the EKS repo's mlflow-probe-timeout-patch.py; kept in
sync deliberately (this repo is self-contained on the cluster network).
"""

import os
import sys

_TARGET = "mlflow.utils.server_info"


def apply(timeout=None, max_retries=None) -> bool:
    """Patch si.http_request (idempotent). True when active.

    No-op (False) when mlflow is not installed or the target module has no
    http_request (very old / refactored mlflow) — keeps --check on bare
    hosts safe.
    """
    try:
        import importlib

        # import_module (not `import ... as`): under py<3.7 the import-as
        # statement getattr's the parent package attribute, which is not set
        # yet when the target module is still mid-import.
        si = importlib.import_module(_TARGET)
    except ImportError:
        return False
    orig = si.http_request
    if getattr(orig, "_mlflow_probe_patch", False):
        return True
    if timeout is None:
        timeout = int(os.environ.get("MLFLOW_PROBE_TIMEOUT", "30"))
    if max_retries is None:
        max_retries = int(os.environ.get("MLFLOW_PROBE_MAX_RETRIES", "3"))

    def _http_request(*args, **kwargs):
        kwargs["timeout"] = timeout
        kwargs["max_retries"] = max_retries
        return orig(*args, **kwargs)

    _http_request._mlflow_probe_patch = True
    si.http_request = _http_request
    return True


def _install_import_hook() -> None:
    """Patch mlflow.utils.server_info right after its first import.

    A .pth import runs in EVERY python process, so this must not import
    mlflow eagerly; the meta-path finder defers all mlflow work until the
    process itself imports the target module, then patches it once and
    removes itself. Marker-based guards: each exec of this file creates a
    fresh finder class, so identity checks are not enough (two copies once
    recursed into each other).
    """
    from importlib.abc import MetaPathFinder

    class _Finder(MetaPathFinder):
        _mlflow_probe_finder = True

        def find_spec(self, fullname, path=None, target=None):
            if fullname != _TARGET:
                return None
            spec = None
            for finder in sys.meta_path:
                if getattr(finder, "_mlflow_probe_finder", False):
                    continue
                find_spec = getattr(finder, "find_spec", None)
                if find_spec is None:
                    continue
                candidate = find_spec(fullname, path)
                if candidate is not None:
                    spec = candidate
                    if candidate.loader is not None:
                        break
            if spec is None or spec.loader is None:
                return None
            sys.meta_path[:] = [
                f for f in sys.meta_path
                if not getattr(f, "_mlflow_probe_finder", False)
            ]
            if isinstance(spec.loader, type):
                return spec
            orig_exec = spec.loader.exec_module

            def _exec_module(module):
                orig_exec(module)
                apply()

            try:
                spec.loader.exec_module = _exec_module
            except AttributeError:
                pass
            return spec

    sys.meta_path[:] = [
        f for f in sys.meta_path if not getattr(f, "_mlflow_probe_finder", False)
    ]
    if _TARGET in sys.modules:
        apply()
        return
    sys.meta_path.insert(0, _Finder())


# Import side effect: register the lazy hook (used by the .pth install; the
# CLIs in this package call apply() directly).
_install_import_hook()


def _cli():
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--install",
        action="store_true",
        help="install into site-packages (+ .pth): every python process in this environment",
    )
    g.add_argument("--remove", action="store_true", help="undo --install")
    g.add_argument("--check", action="store_true", help="report patch state")
    args = p.parse_args()

    if not (args.install or args.remove or args.check):
        p.print_help()
        return

    import shutil
    import sysconfig
    from pathlib import Path

    sp = Path(sysconfig.get_paths()["purelib"])
    src = Path(__file__).with_name("mlflow_probe_patch.py")
    if args.install:
        pkg = sp / "interpam"
        pkg.mkdir(exist_ok=True)
        init = pkg / "__init__.py"
        if not init.exists():
            init.write_text("")
        dest = pkg / "mlflow_probe_patch.py"
        pth = sp / "mlflow-probe-timeout.pth"
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        # The .pth imports interpam.mlflow_probe_patch; a failing .pth line
        # is silently swallowed by site.py, so the import path must match
        # the installed location exactly.
        pth.write_text("import interpam.mlflow_probe_patch\n")
        print(f"installed: {dest}")
        print(f"installed: {pth}")
    elif args.remove:
        for f in (
            sp / "interpam" / "mlflow_probe_patch.py",
            sp / "mlflow-probe-timeout.pth",
        ):
            if f.exists():
                f.unlink()
                print(f"removed: {f}")
            else:
                print(f"not installed: {f}")
    else:
        try:
            import importlib

            import mlflow  # noqa: F401 (version print below)
        except ImportError:
            print("mlflow: NOT INSTALLED (nothing to patch)")
            return
        try:
            # import_module (not `import ... as`): see the note in apply()
            si = importlib.import_module(_TARGET)
        except ImportError:
            print(
                f"mlflow: {getattr(mlflow, '__version__', '?')} has no "
                "mlflow.utils.server_info (no workspace probe — nothing to patch)"
            )
            return
        patched = getattr(si.http_request, "_mlflow_probe_patch", False)
        print(f"mlflow: {getattr(mlflow, '__version__', '?')}")
        print(f"probe patch on import: {'ACTIVE' if patched else 'inactive'}")
        print(
            f"config: MLFLOW_PROBE_TIMEOUT={os.environ.get('MLFLOW_PROBE_TIMEOUT', '30')} "
            f"MLFLOW_PROBE_MAX_RETRIES={os.environ.get('MLFLOW_PROBE_MAX_RETRIES', '3')}"
        )


if __name__ == "__main__":
    _cli()
 




 ### Template 3

 import argparse

import mlflow
from mlflow.environment_variables import MLFLOW_RUN_ID
from mlflow.tracking import MlflowClient

from interpam.mlflow_probe_patch import apply as patch_probe_timeout


def getargs():
    parser = argparse.ArgumentParser(
        description="terminate the mlflow run created by create_mlflow_run "
        "(final step of a multi-step flow sharing MLFLOW_RUN_ID)"
    )

    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        required=False,
        help="mlflow tracking server uri (defaults to MLFLOW_TRACKING_URI env var)",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        required=False,
        help="run id to terminate (defaults to MLFLOW_RUN_ID env var)",
    )

    parser.add_argument(
        "--status",
        type=str,
        default="FINISHED",
        required=False,
        choices=["FINISHED", "FAILED", "KILLED"],
        help="final status of the run",
    )

    return parser.parse_args()


def main():
    args = getargs()

    patch_probe_timeout()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    run_id = args.run_id or MLFLOW_RUN_ID.get()

    if not run_id:
        raise SystemExit(
            "no run id: pass --run-id or set the MLFLOW_RUN_ID environment variable"
        )

    client = MlflowClient()
    client.set_terminated(run_id, status=args.status)
    run = client.get_run(run_id)
    print(f"MLFLOW_RUN_ID={run_id}")
    print(f"run status: {run.info.status}")


if __name__ == "__main__":
    main()
 










