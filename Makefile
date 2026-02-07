# Narra — Narrative intelligence engine
# CLI + MCP server

BINARY    := narra
RELEASE    := target/release/$(BINARY)
DEBUG      := target/debug/$(BINARY)
TEST_DATA  := /tmp/narra-test-data

.PHONY: help build release check test test-unit test-integration test-snapshot \
        clippy fmt fmt-check lint clean doc install size \
        quick-check test-embedding test-full machete pre-release install-hooks

## —— General ——————————————————————————————————————————

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

## —— Build ————————————————————————————————————————————

build: ## Build debug binary
	cargo build

release: ## Build optimised release binary
	cargo build --release

check: ## Type-check without building
	cargo check

## —— Quality ——————————————————————————————————————————

test: ## Run all tests (unit + integration)
	cargo test

test-unit: ## Run unit tests only
	cargo test --lib

test-integration: ## Run integration tests only
	cargo test --test '*'

test-snapshot: ## Update insta snapshots then review
	cargo insta test --review

clippy: ## Run clippy lints
	cargo clippy -- -D warnings

fmt: ## Format code
	cargo fmt

fmt-check: ## Check formatting (CI-friendly)
	cargo fmt -- --check

lint: clippy fmt-check ## Run all lints (clippy + format check)

quick-check: fmt-check clippy ## Fast local check (pre-commit hook)

test-embedding: ## Run embedding model tests (ignored by default)
	cargo test -- --ignored

test-full: test test-embedding ## Full test suite including embedding tests

machete: ## Check for unused dependencies
	cargo machete

pre-release: fmt clippy test-full machete release ## Full validation before tagging a release

## —— Docs —————————————————————————————————————————————

doc: ## Build and open rustdoc
	cargo doc --open --no-deps

## —— Install / Deploy —————————————————————————————————

install: release ## Build release and confirm binary is ready
	@test -f $(RELEASE) && echo "$(RELEASE) ready ($(shell du -h $(RELEASE) | cut -f1))" || (echo "Build failed"; exit 1)

## —— Utilities ————————————————————————————————————————

size: release ## Show release binary size
	@du -h $(RELEASE)

clean: ## Remove build artifacts and test data
	cargo clean
	rm -rf $(TEST_DATA)

install-hooks: ## Install git pre-commit hook
	cp scripts/pre-commit .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit
	@echo "Pre-commit hook installed."
