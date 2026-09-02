#!/usr/bin/env python3
"""Run the packaged upstream-provenance verifier from a protected base checkout."""

from pr_agent.upstream_provenance import main

if __name__ == "__main__":
    raise SystemExit(main())
