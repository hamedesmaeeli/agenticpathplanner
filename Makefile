PYTHON := python3.12
PACKAGES := core benchmarks classical llm evaluation refiner generator selector

.PHONY: install install-dev test test-all lint clean

## Install all packages in editable mode (dev deps included)
install-dev:
	@for pkg in $(PACKAGES); do \
	  echo ">>> Installing packages/$$pkg"; \
	  $(PYTHON) -m pip install -e "packages/$$pkg[dev]" -q; \
	done

## Install all packages (no dev deps)
install:
	@for pkg in $(PACKAGES); do \
	  $(PYTHON) -m pip install -e "packages/$$pkg" -q; \
	done

## Run tests for a single package:  make test PKG=core
test:
	$(PYTHON) -m pytest packages/$(PKG)/tests/ -v

## Run every package's test suite
test-all:
	@for pkg in $(PACKAGES); do \
	  echo "\n=== Testing packages/$$pkg ==="; \
	  $(PYTHON) -m pytest packages/$$pkg/tests/ -v --tb=short || exit 1; \
	done

## Lint all source files
lint:
	$(PYTHON) -m ruff check packages/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
