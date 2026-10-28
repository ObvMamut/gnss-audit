.PHONY: check check-full review figures image-patched

# Fast gate: everything except slow Monte Carlo and Docker tests (docs/11 section 2).
check:
	uv run ruff format --check .
	uv run ruff check .
	uv run pyright
	uv run pytest -m "not slow and not container and not data"

check-full:
	uv run ruff format --check .
	uv run ruff check .
	uv run pyright
	uv run pytest

# Regenerate the review results that nothing pins: s01 to s16, s27 and s28. The frozen records s29 to s35 run only when
# named (review/run_all.py; docs/14 section 2).
review:
	uv run python -m review.run_all

# Redraw every figure into figures/: f14-* from review/results, and the E2 figures f10-* from the pinned run files
# committed in run-summaries/ (D-48). Each source is checked by run id and SHA-256 before anything is drawn.
figures:
	uv run python -I review/figures.py
	GNSSAUDIT_RESULTS=run-summaries uv run python -I review/figures.py g3
	GNSSAUDIT_RESULTS=run-summaries uv run python -I review/figures.py p2c

# Build the patched GNSS-SDR image p2 (D-08, D-27, D-40; containers/Dockerfile.patched) and print its image ID for
# containers/digests.md. Needs network access (GitHub) and about 8 GB of Docker space. MAKE_JOBS=6 limits the
# compile's parallelism (default: every CPU).
MAKE_JOBS ?=
image-patched:
	docker build $(if $(MAKE_JOBS),--build-arg MAKE_JOBS=$(MAKE_JOBS) )-f containers/Dockerfile.patched -t gnssaudit/gnss-sdr:v0.0.22-p2 containers
	docker image inspect --format '{{.Id}}' gnssaudit/gnss-sdr:v0.0.22-p2
