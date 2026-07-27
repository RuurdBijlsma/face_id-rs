set windows-shell := ["powershell.exe", "-Command"]

# --- Lints:

check: fmt clippy test

fmt:
    cargo fmt --all

clippy:
    cargo clippy --no-deps --all-features --tests --benches -- \
        -D clippy::all \
        -D clippy::pedantic \
        -D clippy::nursery

# --- Misc:

clean:
    cargo clean

# --- Execution:

test:
    cargo test --profile release -- --nocapture

bench:
    cargo bench --features cuda

run:
    cargo run --example basic_use --profile release

