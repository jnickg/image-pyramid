// This file is part of the `image-pyramid` crate: <https://github.com/jnickg/image-pyramid>
// Copyright (C) 2024 jnickg <jnickg83@gmail.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.
//
// Copyright (C) 2024 jnickg <jnickg83@gmail.com>
// SPDX-License-Identifier: GPL-3.0-only

#![doc(html_root_url = "https://docs.rs/image-pyramid/0.2.2")]
#![doc(issue_tracker_base_url = "https://github.com/jnickg/image-pyramid/issues")]

//! # Image Pyramid
//!
//! ![Maintenance](https://img.shields.io/badge/maintenance-actively--developed-brightgreen.svg)
//! [![crates-io](https://img.shields.io/crates/v/image-pyramid.svg)](https://crates.io/crates/image-pyramid)
//! [![api-docs](https://docs.rs/image-pyramid/badge.svg)](https://docs.rs/image-pyramid)
//! [![dependency-status](https://deps.rs/repo/github/jnickg/image-pyramid/status.svg)](https://deps.rs/repo/github/jnickg/image-pyramid)
//!
//! ## Overview
//!
//! This is a small Rust crate that facilitates quickly generating an image
//! pyramid from a user-provided image.
//!
//! - See [OpenCV: Image Pyramids](https://docs.opencv.org/4.x/dc/dff/tutorial_py_pyramids.html)
//!   for an overview of the two most common pyramid types, Lowpass (AKA
//!   Gaussian) and Bandpass (AKA Laplacian).
//! - The Tomasi paper [Lowpass and Bandpass Pyramids](https://courses.cs.duke.edu/cps274/fall14/notes/Pyramids.pdf)
//!   has an authoritative explanation as well.
//! - [Wikipedia](https://en.wikipedia.org/wiki/Pyramid_(image_processing)#Steerable_pyramid)
//!   has a decent explanation of a steerable pyramid
//!
//! ## Usage
//!
//! See the [crates.io page](https://crates.io/crates/image-pyramid) for installation instructions, then check out the [examples directory](./examples/) for example code. Below is a simple illustrative example of computing a default pyramid (Gaussian where each level is half resolution).
//!
//! ```rust
//! use image::DynamicImage;
//! use image_pyramid::*;
//!
//! let image = DynamicImage::new_rgba8(640, 480); // Or load from file
//! let pyramid = match ImagePyramid::create(&image, None) {
//!   Ok(pyramid) => pyramid,
//!   Err(e) => {
//!     eprintln!("Error creating image pyramid: {}", e);
//!     return;
//!   }
//! };
//! ```
//!
//! ## Support
//!
//! Open an Issue with questions or bug reports, and feel free to open a PR with
//! proposed changes.
//!
//! ## Contributing
//!
//! Follow standard Rust conventions, and be sure to add tests for any new code
//! added.

#![deny(
  warnings,
  nonstandard_style,
  unused,
  future_incompatible,
  rust_2018_idioms,
  unsafe_code,
  clippy::all,
  clippy::nursery,
  clippy::pedantic
)]
#![recursion_limit = "128"]

use image::{DynamicImage, GenericImage, GenericImageView, Pixel};

/// A simple wrapper extending the functionality of the given image with
/// image-pyramid support
pub struct ImageToProcess<'a>(pub &'a DynamicImage);

/// How to smooth an image when downsampling
///
/// For now, these all use a 3x3 kernel for smoothing. As a consequence, the
/// Gaussian and Triangle smoothing types produce identical results
pub enum SmoothingType {
  /// Use a Gaussian filter
  /// `[[1,2,1],[2,4,2],[1,2,1]] * 1/16`
  Gaussian,
  /// Use a linear box filter
  /// `[[1,1,1],[1,1,1],[1,1,1]] * 1/9`
  Box,
  /// Use a linear triangle filter:
  /// `[[1,2,1],[2,4,2],[1,2,1]] * 1/16`
  Triangle,
}

impl Clone for SmoothingType {
  fn clone(&self) -> Self {
    match self {
      Self::Gaussian => Self::Gaussian,
      Self::Box => Self::Box,
      Self::Triangle => Self::Triangle,
    }
  }
}

/// What type of pyramid to compute. Each has different properties,
/// applications, and computation cost.
pub enum ImagePyramidType {
  /// Use smoothing & subsampling to compute pyramid. This is used to generate
  /// mipmaps, thumbnails, display low-resolution previews of expensive image
  /// processing operations, texture synthesis, and more.
  Lowpass,

  /// AKA Laplacian pyramid, where adjacent levels of the lowpass pyramid are
  /// upscaled and their pixel differences are computed. This used in image
  /// processing routines such as blending.
  Bandpass,

  /// Uses a bank of multi-orientation bandpass filters. Used for used for
  /// applications including image compression, texture synthesis, and object
  /// recognition.
  Steerable,
}

impl Clone for ImagePyramidType {
  fn clone(&self) -> Self {
    match self {
      Self::Lowpass => Self::Lowpass,
      Self::Bandpass => Self::Bandpass,
      Self::Steerable => Self::Steerable,
    }
  }
}

/// The set of parameters required for computing an image pyramid. For most
/// applications, the default set of parameters is correct.
pub struct ImagePyramidParams {
  /// The scale factor to use on image dimensions when downsampling. This is
  /// most commonly 0.5
  pub scale_factor: f32,

  /// What type of pyramid to compute. See [`ImagePyramidType`] for more
  /// information.
  pub pyramid_type: ImagePyramidType,

  /// What type of smoothing to use when computing pyramid levels. See
  /// [`SmoothingType`] for more information.
  pub smoothing_type: SmoothingType,
}

/// Generates a useful default set of parameters.
///
/// Defaults to a traditional image pyramid: Gaussian lowpass image pyramid with
/// scale factor of 0.5.
impl Default for ImagePyramidParams {
  fn default() -> Self {
    Self {
      scale_factor:   0.5,
      pyramid_type:   ImagePyramidType::Lowpass,
      smoothing_type: SmoothingType::Gaussian,
    }
  }
}

impl Clone for ImagePyramidParams {
  fn clone(&self) -> Self {
    Self {
      scale_factor:   self.scale_factor,
      pyramid_type:   self.pyramid_type.clone(),
      smoothing_type: self.smoothing_type.clone(),
    }
  }
}

/// A computed image pyramid and its associated metadata.
///
/// Image pyramids consist of multiple, successively smaller-scale versions of
/// an original image. These are called the _levels_ (sometimes called
/// _octaves_) of the image pyramid.
///
/// Closely related to a traditional Gaussian image pyramid is a mipmap, which
/// is a specific application of the more general image pyramid concept. A
/// mipmap is essentially a way of storing a `scale=0.5` lowpass image pyramid
/// such that an appropriate octave can be sampled by a graphics renderer, for
/// the purpose of avoiding anti-aliasing.
pub struct ImagePyramid {
  /// The ordered levels of the pyramid. Index N refers to pyramid level N.
  /// Depending on the scale factor S in `params`, and image dimensions `(W,
  /// H)`, there will be `ceil(log_{1/S}(min(W, H)))` levels.
  ///
  /// For example, a `(800, 600)` image with scale factor `S=0.5` will have
  /// `ceil(log_2(600))` levels, which comes out to `10`. Similarly, a `(640,
  /// 480)` image would have `(ceil(log_2(480))` (`9`) levels.
  pub levels: Vec<DynamicImage>,

  /// A copy of the parameters used to compute the levels in this pyramid.
  pub params: ImagePyramidParams,
}

impl ImagePyramid {
  /// Create a new image pyramid for the given image, using the optionally
  /// provided parameters.
  ///
  /// If no parameters are passed, the default parameters will be used.
  ///
  /// # Errors
  /// See [`CanComputePyramid::compute_image_pyramid`] for errors that may be
  /// raised
  pub fn create(
    image: &DynamicImage,
    params: Option<&ImagePyramidParams>,
  ) -> Result<Self, &'static str> {
    let image_to_process = ImageToProcess(image);
    let pyramid = image_to_process.compute_image_pyramid(params)?;

    Ok(pyramid)
  }
}

/// Describes types that can compute their own image pyramid
pub trait CanComputePyramid {
  /// Compute an image pyramid for this instance's data, using the optionally
  /// provided parameters.
  ///
  /// If no parameters are passed, the default parameters will be used.
  ///
  /// # Errors
  /// Errors are raised for the following parameter values, which are not yet
  /// implemented:
  ///
  /// - [`SmoothingType::Box`] - This smoothing type is not supported in the
  ///   `image` crate and is not yet implemented manually
  /// - [`ImagePyramidType::Steerable`] - Not yet implemented
  fn compute_image_pyramid(
    &self,
    params: Option<&ImagePyramidParams>,
  ) -> Result<ImagePyramid, &'static str>;
}

impl<'a> CanComputePyramid for ImageToProcess<'a> {
  fn compute_image_pyramid(
    &self,
    params: Option<&ImagePyramidParams>,
  ) -> Result<ImagePyramid, &'static str> {
    /// Compute a lowpass pyramid with the given params. Ignores
    /// `params.pyramid_type`.
    fn compute_lowpass_pyramid(
      image: &DynamicImage,
      params: &ImagePyramidParams,
    ) -> Vec<DynamicImage> {
      let mut levels = Vec::new();
      levels.push(image.clone());
      let filter_type: image::imageops::FilterType = match params.smoothing_type {
        SmoothingType::Gaussian => image::imageops::FilterType::Gaussian,
        SmoothingType::Box => std::unimplemented!(),
        SmoothingType::Triangle => image::imageops::FilterType::Triangle,
      };
      let mut current_level = image.clone();
      #[allow(clippy::cast_possible_truncation)]
      #[allow(clippy::cast_precision_loss)]
      #[allow(clippy::cast_sign_loss)]
      while current_level.width() > 1 && current_level.height() > 1 {
        current_level = current_level.resize_exact(
          (current_level.width() as f32 * params.scale_factor) as u32,
          (current_level.height() as f32 * params.scale_factor) as u32,
          filter_type,
        );
        levels.push(current_level.clone());
      }
      levels
    }

    /// Takes the diference in pixel values between `image` and `other`, adds
    /// that value to the center of the Subpixel container type's range, and
    /// applies the result to `image`.
    fn bandpass_in_place<I>(image: &mut I, other: &I)
    where I: GenericImage {
      use image::Primitive;
      use num_traits::NumCast;
      type Subpixel<I> = <<I as GenericImageView>::Pixel as Pixel>::Subpixel;
      let mid_val = ((Subpixel::<I>::DEFAULT_MAX_VALUE - Subpixel::<I>::DEFAULT_MIN_VALUE)
        / NumCast::from(2).unwrap())
        + Subpixel::<I>::DEFAULT_MIN_VALUE;
      // dbg!(<f32 as NumCast>::from(mid_val).unwrap());
      debug_assert_eq!(image.dimensions(), other.dimensions());
      // Iterate through pixels and compute difference. Add difference to
      // mid_val and apply that to i1
      let (width, height) = image.dimensions();
      for y in 0..height {
        for x in 0..width {
          let other_p = other.get_pixel(x, y);
          let mut p = image.get_pixel(x, y);
          p.apply2(&other_p, |b1, b2| {
            let diff = <f32 as NumCast>::from(b1).unwrap() - <f32 as NumCast>::from(b2).unwrap();
            let new_val = <f32 as NumCast>::from(mid_val).unwrap() + diff;
            // dbg!((diff, new_val));
            NumCast::from(new_val).unwrap_or(mid_val)
          });
          image.put_pixel(x, y, p);
        }
      }
    }

    // If unspecified, use default parameters.
    let params = params.map_or_else(ImagePyramidParams::default, std::clone::Clone::clone);

    match params.pyramid_type {
      ImagePyramidType::Lowpass =>
        match params.smoothing_type {
          SmoothingType::Box => Err("Box filter not yet implemented"),
          _ =>
            Ok(ImagePyramid {
              levels: compute_lowpass_pyramid(self.0, &params),
              params: params.clone(),
            }),
        },
      ImagePyramidType::Bandpass => {
        // First, we need a lowpass pyramid to work with.
        let mut levels = match params.smoothing_type {
          SmoothingType::Box => return Err("Box filter not yet implemented"),
          _ => compute_lowpass_pyramid(self.0, &params),
        };

        // For each index N, upscale the resolution N+1 to match N's resolution.
        // Then we compute the pixel-wise difference between them, and
        // store the result in the current level
        for i in 0..levels.len() - 1 {
          let next_level = levels[i + 1].resize_exact(
            levels[i].width(),
            levels[i].height(),
            image::imageops::FilterType::Nearest,
          );
          bandpass_in_place(&mut levels[i], &next_level);
        }

        Ok(ImagePyramid {
          levels,
          params,
        })
      }
      ImagePyramidType::Steerable => Err("Steerable pyramid not yet implemented"),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn image_pyramid_create_rgb8_800x600() {
    let image = DynamicImage::new_rgb8(800, 600);

    let pyramid = ImagePyramid::create(&image, None).unwrap();
    assert_eq!(pyramid.levels.len(), 10);
  }

  #[test]
  fn image_pyramid_create_rgb8() {
    let image = DynamicImage::new_rgb8(640, 480);

    let pyramid = ImagePyramid::create(&image, None).unwrap();
    assert_eq!(pyramid.levels.len(), 9);
  }

  #[test]
  fn compute_image_pyramid_rgb8() {
    let image = DynamicImage::new_rgb8(640, 480);
    let ipr = ImageToProcess(&image);

    let pyramid = ipr.compute_image_pyramid(None).unwrap();
    assert_eq!(pyramid.levels.len(), 9);
  }

  #[test]
  fn image_pyramid_create_rgba8() {
    let image = DynamicImage::new_rgba8(640, 480);

    let pyramid = ImagePyramid::create(&image, None).unwrap();
    assert_eq!(pyramid.levels.len(), 9);
  }

  #[test]
  fn compute_image_pyramid_rgba8() {
    let image = DynamicImage::new_rgba8(640, 480);
    let ipr = ImageToProcess(&image);

    let pyramid = ipr.compute_image_pyramid(None).unwrap();
    assert_eq!(pyramid.levels.len(), 9);
  }

  #[test]
  fn image_pyramid_create_bandpass_rgb8() {
    let image = DynamicImage::new_rgb8(640, 480);

    let params = ImagePyramidParams {
      pyramid_type: ImagePyramidType::Bandpass,
      ..Default::default()
    };

    let pyramid = ImagePyramid::create(&image, Some(&params)).unwrap();
    assert_eq!(pyramid.levels.len(), 9);
  }

  #[test]
  fn compute_image_pyramid_bandpass_rgb8() {
    let image = DynamicImage::new_rgb8(640, 480);
    let ipr = ImageToProcess(&image);

    let params = ImagePyramidParams {
      pyramid_type: ImagePyramidType::Bandpass,
      ..Default::default()
    };

    let pyramid = ipr.compute_image_pyramid(Some(&params)).unwrap();
    assert_eq!(pyramid.levels.len(), 9);
  }

  #[test]
  fn image_pyramid_create_bandpass_rgba8() {
    let image = DynamicImage::new_rgba8(640, 480);

    let params = ImagePyramidParams {
      pyramid_type: ImagePyramidType::Bandpass,
      ..Default::default()
    };

    let pyramid = ImagePyramid::create(&image, Some(&params)).unwrap();
    assert_eq!(pyramid.levels.len(), 9);
  }

  #[test]
  fn compute_image_pyramid_bandpass_rgba8() {
    let image = DynamicImage::new_rgba8(640, 480);
    let ipr = ImageToProcess(&image);

    let params = ImagePyramidParams {
      pyramid_type: ImagePyramidType::Bandpass,
      ..Default::default()
    };

    let pyramid = ipr.compute_image_pyramid(Some(&params)).unwrap();
    assert_eq!(pyramid.levels.len(), 9);
  }
}
