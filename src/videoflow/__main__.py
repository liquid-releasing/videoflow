"""Entry point for `python -m videoflow`.

Lets the bridge call `python -m videoflow <subcommand> ...` instead of
the pip-installed `videoflow.exe` console-script wrapper. The wrapper
doesn't reliably forward stderr when spawned from non-shell parents
(Tokio's `Command` on Windows), so going through `-m` keeps progress
streaming working.
"""

from videoflow.cli import main

if __name__ == "__main__":
    main()
