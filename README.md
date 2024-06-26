# Image Pyramid

![Maintenance](https://img.shields.io/badge/maintenance-actively--developed-brightgreen.svg)
[![crates-io](https://img.shields.io/crates/v/image-pyramid.svg)](https://crates.io/crates/image-pyramid)
[![api-docs](https://docs.rs/image-pyramid/badge.svg)](https://docs.rs/image-pyramid)
[![dependency-status](https://deps.rs/repo/github/jnickg/image-pyramid/status.svg)](https://deps.rs/repo/github/jnickg/image-pyramid)

## Overview

This is a small Rust crate that facilitates quickly generating an image pyramid from a user-provided image.

- See [OpenCV: Image Pyramids](https://docs.opencv.org/4.x/dc/dff/tutorial_py_pyramids.html) for an overview of the two most common pyramid types, Lowpass (AKA Gaussian) and Bandpass (AKA Laplacian).
- The Tomasi paper [Lowpass and Bandpass Pyramids](https://courses.cs.duke.edu/cps274/fall14/notes/Pyramids.pdf) has an authoritative explanation as well.
- [Wikipedia](https://en.wikipedia.org/wiki/Pyramid_(image_processing)#Steerable_pyramid) has a decent explanation of a steerable pyramid

## Usage

See the [crates.io page](https://crates.io/crates/image-pyramid) for installation instructions, then check out the [examples directory](./examples/) for example code. Below is a simple illustrative example of computing a default pyramid (Gaussian where each level is half resolution).

```rust
use image_pyramid::*;

let image = todo!();
let pyramid = match ImagePyramid::create(&image, None) {
    Ok(pyramid) => pyramid,
    Err(e) => {
        eprintln!("Error creating image pyramid: {}", e);
        return;
    }
};
```

Or a slightly more complex example, illustrating how to create a bandpass pyramid where each octave is $2\over{3}$ the resolution, smoothed using a triangle (linear) filter.

```rust
use image_pyramid::*;

let image = todo!();
let params = ImagePyramidParams {
    scale_factor: (2.0 / 3.0).into_unit_interval().unwrap(),
    pyramid_type: ImagePyramidType::Bandpass,
    smoothing_type: SmoothingType::Triangle
};
let pyramid = match ImagePyramid::create(&image, Some(&params)) {
    Ok(pyramid) => pyramid,
    Err(e) => {
        eprintln!("Error creating image pyramid: {}", e);
        return;
    }
};
```

The `scale_factor` field is a `UnitIntervalValue`, which must be in the interval $(0, 1)$. Creating a value of this type yields a `Result` and will contain an error if the value is not valid.

## Support

Open an Issue with questions or bug reports, and feel free to open a PR with proposed changes.

## Contributing

Follow standard Rust conventions, and be sure to add tests for any new code added.
