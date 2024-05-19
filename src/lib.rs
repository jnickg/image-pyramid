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

#![doc(html_root_url = "https://docs.rs/image-pyramid/0.4.1")]
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
//! Or a slightly more complex example, illustrating how to create a bandpass
//! pyramid where each octave is 2/3 the resolution, smoothed using a triangle
//! (linear) filter.
//!
//! ```rust
//! use image::DynamicImage;
//! use image_pyramid::*;
//!
//! let image = DynamicImage::new_rgba8(640, 480); // Or load from file
//! let params = ImagePyramidParams {
//!   scale_factor:   (2.0 / 3.0).into_unit_interval().unwrap(),
//!   pyramid_type:   ImagePyramidType::Bandpass,
//!   smoothing_type: SmoothingType::Triangle,
//! };
//! let pyramid = match ImagePyramid::create(&image, Some(&params)) {
//!   Ok(pyramid) => pyramid,
//!   Err(e) => {
//!     eprintln!("Error creating image pyramid: {}", e);
//!     return;
//!   }
//! };
//! ```
//!
//! [`ImagePyramidParams::scale_factor`] field is a [`UnitIntervalValue`], which
//! must be a floating-point value in the interval (0, 1). Creating a value of
//! this type yields a [`Result`] and will contain an error if the value is not
//! valid.
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
  nonstandard_style,
  unused,
  future_incompatible,
  rust_2018_idioms,
  unsafe_code,
  clippy::all,
  clippy::nursery,
  clippy::pedantic
)]

use image::{DynamicImage, GenericImage, GenericImageView, Pixel};
use num_traits::NumCast;
use thiserror::Error;

/// An enumeration of the errors that may be emitted from the `image_pyramid`
/// crate
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum ImagePyramidError {
  /// Raised when the user provides an invalid scale value
  #[error("Invalid scale_factor value {0} (expected: 0.0 < scale_factor < 1.0)")]
  BadScaleFactor(f32),

  /// Raised when the requested functionality is not yet supported.
  #[error("Functionality \"{0}\" is not yet implemented.")]
  NotImplemented(String),

  /// Raised when something unexpected went wrong in the library.
  #[error("Internal error: {0}")]
  Internal(String),
}

/// A container for a value falling on the range (0.0, 1.0) (exclusive, meaning
/// the values 0.0 and 1.0 are not valid)
///
/// This is useful for safely defining decreasing scale factors.
///
/// Because this type is currently used only for computing the resized
/// dimensions of each level of the pyramid, the choice was made to support only
/// [`f32`]. In the future support may be added for other floating-point types,
/// such as [`f64`], rational values, or fixed-point, if there arises a need for
/// such precision and/or performance.
#[derive(Debug, Copy, Clone)]
pub struct UnitIntervalValue(f32);

/// A trait describing some floating-point type that can be converted to a
/// unit-interval value (0.0 to 1.0, exclusive)
pub trait IntoUnitInterval {
  /// Attempts to convert this value into a guaranteed unit-interval value.
  ///
  /// Returns an error string if the value is not valid.
  ///
  /// # Errors
  /// - The value is not within the unit range
  fn into_unit_interval(self) -> Result<UnitIntervalValue, ImagePyramidError>;
}

impl IntoUnitInterval for f32 {
  fn into_unit_interval(self) -> Result<UnitIntervalValue, ImagePyramidError> {
    match self {
      v if v <= 0.0 || v >= 1.0 => Err(ImagePyramidError::BadScaleFactor(v)),
      _ => Ok(UnitIntervalValue(self)),
    }
  }
}

impl UnitIntervalValue {
  /// Attempts to create a new instance from the provided value
  ///
  /// # Errors
  /// - The value is not within the unit range
  pub fn new<T: IntoUnitInterval>(val: T) -> Result<Self, ImagePyramidError> {
    val.into_unit_interval()
  }

  /// Retrieves the stored value which is guaranteed to fall between 0.0 and 1.0
  /// (exclusive)
  #[must_use]
  pub const fn get(self) -> f32 { self.0 }
}

/// A simple wrapper extending the functionality of the given image with
/// image-pyramid support
pub struct ImageToProcess<'a>(pub &'a DynamicImage);

/// How to smooth an image when downsampling
///
/// For now, these all use a 3x3 kernel for smoothing. As a consequence, the
/// Gaussian and Triangle smoothing types produce identical results
#[derive(Debug, Clone)]
#[non_exhaustive]
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

/// What type of pyramid to compute. Each has different properties,
/// applications, and computation cost.
#[derive(Debug, Clone)]
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

/// The set of parameters required for computing an image pyramid. For most
/// applications, the default set of parameters is correct.
#[derive(Debug, Clone)]
pub struct ImagePyramidParams {
  /// The scale factor to use on image dimensions when downsampling. This is
  /// most commonly 0.5
  pub scale_factor: UnitIntervalValue,

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
      scale_factor:   UnitIntervalValue::new(0.5).unwrap(),
      pyramid_type:   ImagePyramidType::Lowpass,
      smoothing_type: SmoothingType::Gaussian,
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
  ) -> Result<Self, ImagePyramidError> {
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
  /// Errors of type [`ImagePyramidError::NotImplemented`] are raised for the
  /// following parameter values, which are not yet implemented:
  ///
  /// - [`SmoothingType::Box`] - This smoothing type is not yet supported in the
  ///   `image` crate and is also not yet implemented manually
  /// - [`ImagePyramidType::Steerable`] - Not yet implemented
  fn compute_image_pyramid(
    &self,
    params: Option<&ImagePyramidParams>,
  ) -> Result<ImagePyramid, ImagePyramidError>;
}

impl<'a> CanComputePyramid for ImageToProcess<'a> {
  fn compute_image_pyramid(
    &self,
    params: Option<&ImagePyramidParams>,
  ) -> Result<ImagePyramid, ImagePyramidError> {
    /// Compute a lowpass pyramid with the given params. Ignores
    /// `params.pyramid_type`.
    fn compute_lowpass_pyramid(
      image: &DynamicImage,
      params: &ImagePyramidParams,
    ) -> Result<Vec<DynamicImage>, ImagePyramidError> {
      let mut levels = vec![image.clone()];
      let filter_type: image::imageops::FilterType = match params.smoothing_type {
        SmoothingType::Gaussian => image::imageops::FilterType::Gaussian,
        SmoothingType::Box =>
          return Err(ImagePyramidError::NotImplemented(
            "SmoothingType::Box".to_string(),
          )),
        SmoothingType::Triangle => image::imageops::FilterType::Triangle,
      };
      let mut current_level = image.clone();
      #[allow(clippy::cast_possible_truncation)]
      #[allow(clippy::cast_precision_loss)]
      #[allow(clippy::cast_sign_loss)]
      while current_level.width() > 1 && current_level.height() > 1 {
        current_level = current_level.resize_exact(
          (current_level.width() as f32 * params.scale_factor.get()) as u32,
          (current_level.height() as f32 * params.scale_factor.get()) as u32,
          filter_type,
        );
        levels.push(current_level.clone());
      }
      Ok(levels)
    }

    /// Takes the diference in pixel values between `image` and `other`, adds
    /// that value to the center of the Subpixel container type's range, and
    /// applies the result to `image`.
    fn bandpass_in_place<I>(image: &mut I, other: &I)
    where I: GenericImage {
      use image::Primitive;
      type Subpixel<I> = <<I as GenericImageView>::Pixel as Pixel>::Subpixel;
      // The center value for the given container type. We leverage the `image`
      // crate's definition of these values to be 1.0 and 0.0 respectively, for
      // floating-point types (where we want `mid_val` to be 0.5). For unsigned
      // integer types, we should get half the primitive container's capacity
      // (e.g. 127 for a u8)
      let mid_val = ((Subpixel::<I>::DEFAULT_MAX_VALUE - Subpixel::<I>::DEFAULT_MIN_VALUE)
        / NumCast::from(2).unwrap())
        + Subpixel::<I>::DEFAULT_MIN_VALUE;
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
        Ok(ImagePyramid {
          levels: compute_lowpass_pyramid(self.0, &params)?,
          params: params.clone(),
        }),
      ImagePyramidType::Bandpass => {
        // First, we need a lowpass pyramid to work with.
        let mut levels = compute_lowpass_pyramid(self.0, &params)?;

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
      ImagePyramidType::Steerable =>
        Err(ImagePyramidError::NotImplemented(
          "ImagePyramidType::Steerable".to_string(),
        )),
    }
  }
}

#[cfg(test)]
mod tests {
  use test_case::test_matrix;

  use super::*;

  #[test]
  fn compute_image_pyramid_smoothingtype_box_unimplemented() {
    let image = DynamicImage::new_rgb8(640, 480);
    let ipr = ImageToProcess(&image);

    let params = ImagePyramidParams {
      smoothing_type: SmoothingType::Box,
      ..Default::default()
    };

    let pyramid = ipr.compute_image_pyramid(Some(&params));
    assert!(pyramid.is_err());
  }

  #[test]
  fn compute_image_pyramid_imagepyramidtype_steerable_unimplemented() {
    let image = DynamicImage::new_rgb8(640, 480);
    let ipr = ImageToProcess(&image);

    let params = ImagePyramidParams {
      pyramid_type: ImagePyramidType::Steerable,
      ..Default::default()
    };

    let pyramid = ipr.compute_image_pyramid(Some(&params));
    assert!(pyramid.is_err());
  }

  #[test_matrix(
    [ImagePyramidType::Lowpass, ImagePyramidType::Bandpass],
    [SmoothingType::Gaussian, SmoothingType::Triangle]
  )]
  fn compute_image_pyramid_every_type(
    pyramid_type: ImagePyramidType,
    smoothing_type: SmoothingType,
  ) {
    // test_case crate won't let these be parameterized so we loop through them
    // here.
    let functors = vec![
      DynamicImage::new_luma16,
      DynamicImage::new_luma8,
      DynamicImage::new_luma_a16,
      DynamicImage::new_luma_a8,
      DynamicImage::new_rgb16,
      DynamicImage::new_rgb8,
      DynamicImage::new_rgb32f,
      DynamicImage::new_rgba16,
      DynamicImage::new_rgba8,
      DynamicImage::new_rgba32f,
    ];
    for functor in functors {
      let image = functor(128, 128);
      let ipr = ImageToProcess(&image);

      let params = ImagePyramidParams {
        pyramid_type: pyramid_type.clone(),
        smoothing_type: smoothing_type.clone(),
        ..Default::default()
      };

      let pyramid = ipr.compute_image_pyramid(Some(&params));
      assert!(pyramid.is_ok());
      let pyramid = pyramid.unwrap();
      assert_eq!(pyramid.levels.len(), 8);
    }
  }

  #[test]
  fn into_unit_interval_f32() {
    let i = 0.5f32.into_unit_interval();
    assert!(i.is_ok());
    assert_eq!(0.5f32, i.unwrap().get());
  }

  #[test]
  fn into_unit_interval_err_when_0_0f32() {
    let i = 0.0f32.into_unit_interval();
    assert!(i.is_err());
  }

  #[test]
  fn into_unit_interval_err_when_1_0f32() {
    let i = 1.0f32.into_unit_interval();
    assert!(i.is_err());
  }
}
