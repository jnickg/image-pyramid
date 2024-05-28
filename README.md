# Image Pyramid

![Maintenance](https://img.shields.io/badge/maintenance-actively--developed-brightgreen.svg)
[![crates-io](https://img.shields.io/crates/v/image-pyramid.svg)](https://crates.io/crates/image-pyramid)
[![api-docs](https://docs.rs/image-pyramid/badge.svg)](https://docs.rs/image-pyramid)
[![dependency-status](https://deps.rs/repo/github/jnickg/image-pyramid/status.svg)](https://deps.rs/repo/github/jnickg/image-pyramid)

## Overview

This is a small Rust crate that facilitates quickly generating an image
pyramid from a user-provided image. It supports

- Lowpass pyramids (sometimes called Gaussian pyramids, or just "image pyramids"). These are the basis for mipmaps.
- Bandpass pyramids (often called Laplacian pyramids)
- Steerable pyramids, which are explained [here](http://www.cns.nyu.edu/~eero/steerpyr/)

For the lowpass and bandpass pyramids, the user can specify the type of smoothing to use when downsampling the image. The default is a Gaussian filter, but a box filter and triangle filter are also available.

The [`image`](https://crates.io/crates/image) crate is used for image I/O and
manipulation, and the [`num-traits`](https://crates.io/crates/num-traits) crate
is used for numeric operations.

## Background & Usage

See the [`crates.io` page](https://crates.io/crates/image-pyramid) for more information on image pyramids in general, as well as usage & documentation.

## Building

All the normal `cargo` commands should work as expected.

Use `RUSTDOCFLAGS="--html-in-header katex-header.html" cargo doc --no-deps` to build documentation with LaTeX support. Additionally, see [this repo](https://github.com/paulkernfeld/rustdoc-katex-demo) for more info on shoehorning in LaTeX support.

## Support

Open an Issue with questions, feature requests, or bug reports, and feel free to open a PR with proposed changes.

## Contributing

Follow standard Rust conventions, and be sure to add tests for any new code added. Any opened PRs should pass `cargo clippy` and have no failing test.

## License

GPLv3. See [`LICENSE`](./LICENSE) for full license.
