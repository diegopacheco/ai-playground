#!/bin/bash
cd "$(dirname "$0")"
cargo build --release
mkdir -p sample
cp -r agents sample/
cp -r skills sample/
cp target/release/rad sample/
cd sample
./rad
