# Install the pinned toolchain and build the crate.
rustup toolchain install 1.85
rustup override set 1.85
cargo build
