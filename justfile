set windows-shell := ["powershell.exe", "-Command"]

# Run all CI-equivalent checks: format check, clippy, and tests
check: fmt clippy test

# Format the code using rustfmt
fmt:
    cargo fmt --all

# Run Clippy with the same strict flags used in CI
clippy:
    cargo clippy --no-deps --all-features --tests --benches -- \
        -D clippy::all \
        -D clippy::pedantic \
        -D clippy::nursery

# Run all tests including integration tests
test:
    cargo test --profile release

# Run benchmarks
bench:
    cargo bench --all-features

# Clean build artifacts
clean:
    cargo clean