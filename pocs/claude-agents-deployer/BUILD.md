# RAD Build Guide

## Optimized Build with Embedded Assets

RAD (Rust Agent Deployer) uses `rust-embed` to embed all files from the `agents/` and `skills/` directories directly into the binary at compile time.

### What This Means

The compiled binary is completely self-contained and does NOT require the `agents/` or `skills/` directories to exist on the filesystem at runtime. All agent and skill files are embedded directly in the executable.

### Building

Run the build script:

```bash
./build.sh
```

This will:
1. Compile with `--release` flag for optimizations
2. Embed all `.md` files from `agents/` directory
3. Embed all files from `skills/` directory
4. Create an optimized binary at `target/release/rad`

### How It Works

The binary uses a fallback mechanism:

1. First, it checks if `agents/` and `skills/` directories exist in the current directory
2. If not found, it uses the embedded assets compiled into the binary
3. This means you can distribute just the binary without any additional files

### Verification

To verify the embedded assets are in the binary:

```bash
./verify-embedded.sh
```

### Distribution

You can copy the binary to any location and it will work:

```bash
cp target/release/rad /usr/local/bin/
rad
```

The binary will use its embedded assets automatically.

### Technical Details

Uses the `rust-embed` crate with the following embedded resources:

```rust
#[derive(RustEmbed)]
#[folder = "agents/"]
struct AgentsAssets;

#[derive(RustEmbed)]
#[folder = "skills/"]
struct SkillsAssets;
```

At compile time, all files are read and embedded as static data in the binary.
