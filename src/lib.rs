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

#![doc(html_root_url = "https://docs.rs/image-pyramid/0.5.1")]
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
//! pyramid from a user-provided image. It supports
//!
//! - Lowpass pyramids (sometimes called Gaussian pyramids, or just "image
//!   pyramids"). These are the basis for mipmaps.
//! - Bandpass pyramids (often called Laplacian pyramids)
//! - Steerable pyramids, which are explained [here](http://www.cns.nyu.edu/~eero/steerpyr/)
//!
//! For the lowpass and bandpass pyramids, the user can specify the type of
//! smoothing to use when downsampling the image. The default is a Gaussian
//! filter, but a box filter and triangle filter are also available.
//!
//! The [`image`](https://crates.io/crates/image) crate is used for image I/O and
//! manipulation, and the [`num-traits`](https://crates.io/crates/num-traits) crate
//! is used for numeric operations.
//!
//! ## Background
//!
//! - See [OpenCV: Image Pyramids](https://docs.opencv.org/4.x/dc/dff/tutorial_py_pyramids.html)
//!   for an overview of the two most common pyramid types, Lowpass (AKA
//!   Gaussian) and Bandpass (AKA Laplacian).
//! - The Tomasi paper [Lowpass and Bandpass Pyramids](https://courses.cs.duke.edu/cps274/fall14/notes/Pyramids.pdf)
//!   has an authoritative explanation as well.
//! - [Wikipedia](https://en.wikipedia.org/wiki/Pyramid_(image_processing)#Steerable_pyramid)
//!   has a decent explanation of a steerable pyramid
//! - [This NYU Page](http://www.cns.nyu.edu/~eero/steerpyr/) has a good
//!   explanation of steerable pyramids
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
//!   scale_factor: (2.0 / 3.0).into_unit_interval().unwrap(),
//!   pyramid_type: ImagePyramidType::Bandpass(SmoothingType::Triangle(OddValue::new(5).unwrap())),
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
  unsafe_code,
  future_incompatible,
  rust_2018_idioms,
  clippy::all,
  clippy::nursery,
  clippy::pedantic
)]
#![allow(
  clippy::similar_names,
  clippy::doc_markdown,
  clippy::cast_lossless,
  clippy::cast_precision_loss
)]

#[cfg(test)]
#[macro_use]
extern crate approx;

use std::fmt::Debug;

use image::{DynamicImage, GenericImage, GenericImageView, Pixel};
use num_traits::{clamp, Num, NumCast};
use thiserror::Error;

/// An enumeration of the errors that may be emitted from the `image_pyramid`
/// crate
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum ImagePyramidError {
  /// Raised when the user provides an invalid scale value
  #[error("Invalid scale_factor value {0} (expected: 0.0 < scale_factor < 1.0)")]
  BadScaleFactor(f32),

  /// Raised when the user provides an invalid kernel size
  #[error("Invalid kernel size {0} (expected: odd)")]
  BadKernelSize(u8),

  /// Raised when the user provides an invalid parameter somewhere that a more specific error
  /// is not available
  #[error("Bad parameter: {0}")]
  BadParameter(String),

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

fn accumulate<P, K>(acc: &mut [K], pixel: &P, weight: K)
where
  P: Pixel,
  <P as Pixel>::Subpixel: Into<K>,
  K: Num + Copy + Debug,
{
  acc
    .iter_mut()
    .zip(pixel.channels().iter())
    .for_each(|(a, c)| {
      let new_val = <<P as Pixel>::Subpixel as Into<K>>::into(*c) * weight;
      *a = *a + new_val;
    });
}

pub struct Kernel<K> {
  data:   Vec<K>,
  width:  u32,
  height: u32,
}

impl<K: Num + Copy + Debug> Kernel<K> {
  /// Construct a kernel from a slice and its dimensions. The input slice is
  /// in row-major form. For example, a 3x3 matrix with data
  /// `[0,1,0,1,2,1,0,1,0`] describes the following matrix:
  ///
  /// $\begin{bmatrix} 0 & 1 & 0 \\\\ 1 & 2 & 1 \\\\ 0 & 1 & 0 \\\\
  /// \end{bmatrix}$
  ///
  /// # Errors
  ///
  /// - If `width == 0 || height == 0`, [`ImagePyramidError::Internal`] is
  ///   raised
  /// - If the provided data does not match the size corresponding to the given
  ///   dimensions
  ///
  /// # Panics
  ///
  /// In debug builds, this factory panics under the conditions that [`Err`] is
  /// returned for release builds.
  pub fn new(data: &[K], width: u32, height: u32) -> Result<Self, ImagePyramidError> {
    // Take the above asserts and return Internal error when appropriate
    if width == 0 || height == 0 {
      return Err(ImagePyramidError::Internal(
        "width and height must be non-zero".to_string(),
      ));
    }
    if (width * height) as usize != data.len() {
      return Err(ImagePyramidError::Internal(format!(
        "Invalid kernel len: expecting {}, found {}",
        width * height,
        data.len()
      )));
    }

    Ok(Self {
      data: data.to_vec(),
      width,
      height,
    })
  }

  /// Construct a kernel from a slice and its dimensions, normalizing the data
  /// to sum to 1.0. The input slice is in row-major form. For example, a 3x3
  /// matrix with data `[0,1,0,1,2,1,0,1,0`] describes the following matrix:
  ///
  /// $\begin{bmatrix} 0 & 1 & 0 \\\\ 1 & 2 & 1 \\\\ 0 & 1 & 0 \\\\ \end{bmatrix} / 6$
  ///
  /// ...where `6` is computed dynamically by summing the elements of the
  /// kernel. In other words, all the weights in a normalized kernel sum to
  /// 1.0. This is useful, as many filters have this property
  ///
  /// # Errors
  ///
  /// - If `width == 0 || height == 0`, [`ImagePyramidError::Internal`] is
  ///   raised
  /// - If the provided data does not match the size corresponding to the given
  ///   dimensions
  ///
  /// # Panics
  ///
  /// In debug builds, this factory panics under the conditions that [`Err`] is
  /// returned for release builds.
  pub fn new_normalized(
    data: &[K],
    width: u32,
    height: u32,
  ) -> Result<Kernel<f32>, ImagePyramidError>
  where
    K: Into<f32>,
  {
    let mut sum = K::zero();
    for i in data {
      sum = sum + *i;
    }
    let data_norm: Vec<f32> = data
      .iter()
      .map(|x| <K as Into<f32>>::into(*x) / <K as Into<f32>>::into(sum))
      .collect();
    Kernel::<f32>::new(&data_norm, width, height)
  }

  pub(crate) fn new_gaussian(kernel_size: &OddValue) -> Result<Kernel<f32>, ImagePyramidError>
  where K: From<f32> + Into<f32> {
    let kernel_size = kernel_size.get() as u32;
    let mut data = vec![K::zero(); (kernel_size * kernel_size) as usize];
    let sigma = 0.3f32.mul_add((kernel_size as f32 - 1.0).mul_add(0.5, -1.0), 0.8);
    for y in 0..kernel_size {
      for x in 0..kernel_size {
        let x_f = x as f32 - (kernel_size as f32) / 2.0;
        let y_f = y as f32 - (kernel_size as f32) / 2.0;
        let val = sample_gaussian_2d(x_f, y_f, sigma);
        let idx = (y * kernel_size + x) as usize;
        data[idx] = val.into();
      }
    }
    Self::new_normalized(&data, kernel_size, kernel_size)
  }

  pub(crate) fn new_triangle(kernel_size: &OddValue) -> Result<Kernel<f32>, ImagePyramidError>
  where K: From<f32> + Into<f32> {
    let kernel_size = kernel_size.get() as u32;
    let mut data = vec![K::zero(); (kernel_size * kernel_size) as usize];
    for y in 0..kernel_size {
      for x in 0..kernel_size {
        let x_f = x as f32 - (kernel_size as f32) / 2.0;
        let y_f = y as f32 - (kernel_size as f32) / 2.0;
        let val = sample_triangle_2d(x_f, y_f, 1.0);
        let idx = (y * kernel_size + x) as usize;
        data[idx] = val.into();
      }
    }
    Self::new_normalized(&data, kernel_size, kernel_size)
  }

  pub(crate) fn new_box(kernel_size: &OddValue) -> Result<Kernel<f32>, ImagePyramidError>
  where K: From<f32> + Into<f32> {
    let kernel_size = kernel_size.get() as u32;
    let mut data = vec![K::zero(); (kernel_size * kernel_size) as usize];
    for y in 0..kernel_size {
      for x in 0..kernel_size {
        let val = 1.0;
        let idx = (y * kernel_size + x) as usize;
        data[idx] = val.into();
      }
    }
    Self::new_normalized(&data, kernel_size, kernel_size)
  }

  /// Returns 2d correlation of an image. Intermediate calculations are
  /// performed at type K, and the results converted to pixel Q via f. Pads by
  /// continuity.
  #[allow(unsafe_code)]
  #[allow(unused)]
  pub fn filter_in_place<I, F>(&self, image: &mut I, mut f: F)
  where
    I: GenericImage + Clone,
    <<I as GenericImageView>::Pixel as Pixel>::Subpixel: Into<K>,
    F: FnMut(&mut <<I as GenericImageView>::Pixel as Pixel>::Subpixel, K),
  {
    use core::cmp::{max, min};
    let (width, height) = image.dimensions();
    let num_channels = <<I as GenericImageView>::Pixel as Pixel>::CHANNEL_COUNT as usize;
    let zero = K::zero();
    let mut acc = vec![zero; num_channels];
    #[allow(clippy::cast_lossless)]
    let (k_width, k_height) = (self.width as i64, self.height as i64);
    #[allow(clippy::cast_lossless)]
    let (width, height) = (width as i64, height as i64);

    for y in 0..height {
      for x in 0..width {
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let x_u32 = x as u32;
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let y_u32 = y as u32;
        for k_y in 0..k_height {
          #[allow(clippy::cast_possible_truncation)]
          #[allow(clippy::cast_sign_loss)]
          let y_p = clamp(y + k_y - k_height / 2, 0, height - 1) as u32;
          for k_x in 0..k_width {
            #[allow(clippy::cast_possible_truncation)]
            #[allow(clippy::cast_sign_loss)]
            let x_p = clamp(x + k_x - k_width / 2, 0, width - 1) as u32;
            #[allow(clippy::cast_possible_truncation)]
            #[allow(clippy::cast_sign_loss)]
            let k_idx = (k_y * k_width + k_x) as usize;

            accumulate(
              &mut acc,
              unsafe { &image.unsafe_get_pixel(x_p, y_p) },
              unsafe { *self.data.get_unchecked(k_idx) },
            );
          }
        }
        let mut out_pel = image.get_pixel(x_u32, y_u32);
        let out_channels = out_pel.channels_mut();
        for (a, c) in acc.iter_mut().zip(out_channels.iter_mut()) {
          f(c, *a);
          *a = zero;
        }
        image.put_pixel(x_u32, y_u32, out_pel);
      }
    }
  }
}

/// A simple wrapper extending the functionality of the given image with
/// image-pyramid support
pub struct ImageToProcess<'a>(pub &'a DynamicImage);

/// How to smooth an image when downsampling
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SmoothingType {
  /// Use a Gaussian filter
  Gaussian(OddValue),
  /// Use a linear box filter
  Box(OddValue),
  /// Use a linear triangle filter:
  Triangle(OddValue),
}

/// A value that is guaranteed to be odd
///
/// This is useful for kernel sizes, which must be odd to have a well-defined
/// center
#[derive(Debug, Clone)]
pub struct OddValue(pub(crate)u8);

impl OddValue {
  /// Attempts to create a new instance from the provided value
  ///
  /// # Errors
  /// - The value is not odd, which is required for kernel sizes. This returns
  ///   [`ImagePyramidError::BadKernelSize`]
  pub const fn new(val: u8) -> Result<Self, ImagePyramidError> {
    if val % 2 == 0 {
      Err(ImagePyramidError::BadKernelSize(val))
    } else {
      Ok(Self(val))
    }
  }

  #[must_use]
  pub const fn get(&self) -> u8 { self.0 }
}

/// A trait describing some integer type that can be converted to an odd value.
///
/// The value must be storable as a u8, as kernel sizes are typically small.
pub trait IntoOddValue {
  /// Converts this value into a guaranteed odd value.
  ///
  /// # Errors
  /// - The value is not odd. This returns [`ImagePyramidError::BadKernelSize`]
  fn into_odd_value(self) -> Result<OddValue, ImagePyramidError>;
}

impl IntoOddValue for u8 {
  fn into_odd_value(self) -> Result<OddValue, ImagePyramidError> { OddValue::new(self) }
}

/// Implementation for the default (undecorated) literal integer type
impl IntoOddValue for i32 {
  fn into_odd_value(self) -> Result<OddValue, ImagePyramidError> {
    let val: u8 = self.try_into().map_err(|_| {
      ImagePyramidError::BadParameter("The given i32 value can't be converted to u8".to_string())
    })?;
    OddValue::new(val)
  }
}

/// Parameters for generating a steerable image pyramid. These parameters are
/// used to compute the steerable filters for each orientation.
///
/// Steerable Filters are a class of oriented filters that can be expressed as a
/// linear combination of a set of basis filters. As an example, let us consider
/// isotropic Gaussian filter $G(x,y)$:
/// - $G(x,y)=e^{−(x2+y2)}$
///
/// First derivative of $G(x,y)$ is given by $G_1$ and let $G^{\theta}_{1}$ be
/// the first derivative rotated by an angle $\theta$ about the origin. In $x$
/// direction the angle $\theta=0^{\circ}$ and in $y$ direction,
/// $\theta=90^{\circ}$. The first derivatives in $x$ and $y$ directions are
/// given by:
/// - $G^{0^{\circ}}_1 = {{\partial{G}}\over{\partial{x}}} = −2xe^{−(x^2+y^2)}$,
///   and
/// - $G^{90^{\circ}}_1 = {{\partial{G}}\over{\partial{y}}} = −2ye^{−(x^2+y^2)}$
///
/// In 2D space $G^{0^{\circ}}_1$ and $G^{90^{\circ}}_1$ are seen to span the
/// entire space and are the basis filters. An arbitrarily oriented first
/// derivative filter can be expressed as a linear combination of these two
/// filters:
///
/// $G^{θ^{\circ}}_1 = G^{0^{\circ}}_1 \cos{\theta} + G^{90^{\circ}}_1
/// \sin{\theta}$
///
/// This structure determines the number of orientations to compute, which are
/// evenly spaced around the unit circle. The kernel size determines the size of
/// the kernel to use for each filter.
#[derive(Debug, Clone)]
pub struct SteerableParams {
  /// The number of orientations to compute.
  ///
  /// This is the number of filters to
  /// compute, each rotated by an angle of `360 / num_orientations` degrees
  /// using the formulas above.
  ///
  /// The default value is `4`, which is the most common value used in practice.
  pub num_orientations: u8,

  /// The size of the kernel to use for each filter. This is the size of the
  /// kernel in both the x and y directions.
  pub kernel_size: OddValue,
}

impl Default for SteerableParams {
  fn default() -> Self {
    Self {
      num_orientations: 4,
      kernel_size:      OddValue(5),
    }
  }
}

pub(crate) fn sample_triangle_1d(x: f32, w: f32) -> f32 { 1.0 - (x / w).abs() }

pub(crate) fn sample_triangle_2d(x: f32, y: f32, w: f32) -> f32 {
  sample_triangle_1d(x, w) * sample_triangle_1d(y, w)
}

pub(crate) fn sample_gaussian_1d(x: f32, sigma: f32) -> f32 {
  ((2.0 * std::f32::consts::PI).sqrt() * sigma).recip() * (-x.powi(2) / (2.0 * sigma.powi(2))).exp()
}

pub(crate) fn sample_gaussian_2d(x: f32, y: f32, sigma: f32) -> f32 {
  sample_gaussian_1d(x, sigma) * sample_gaussian_1d(y, sigma)
}

pub(crate) fn sample_gaussian_1d_derivative(x: f32, sigma: f32) -> f32 {
  -x * sample_gaussian_1d(x, sigma) / sigma.powi(2)
}

pub(crate) fn sample_gaussian_2d_derivative_x(x: f32, y: f32, sigma: f32) -> f32 {
  sample_gaussian_1d_derivative(x, sigma) * sample_gaussian_1d(y, sigma)
}

pub(crate) fn sample_gaussian_2d_derivative_y(x: f32, y: f32, sigma: f32) -> f32 {
  sample_gaussian_1d(x, sigma) * sample_gaussian_1d_derivative(y, sigma)
}

impl SteerableParams {
  #[must_use]
  pub const fn new(num_orientations: u8, kernel_size: OddValue) -> Self {
    Self {
      num_orientations,
      kernel_size,
    }
  }

  /// This takes the number of orientations and computes the first-order
  /// derivative-of-Gaussian kernels for each orientation.
  ///
  /// It starts by computing the angles for each filter, then creates the two
  /// basis filters, and then computes each steerable filter as a linear
  /// combination of the two basis filters.
  pub(crate) fn get_kernels(&self) -> Vec<Kernel<f32>> {
    let angles: Vec<f32> = (0..self.num_orientations)
      .map(|i| (i as f32) * 180.0 / (self.num_orientations as f32))
      .collect();

    angles
      .iter()
      .map(|angle| {
        let angle_rad = angle.to_radians();
        Kernel::new(
          &(0..self.kernel_size.get())
            .flat_map(|y| {
              (0..self.kernel_size.get()).map(move |x| {
                let x_f = x as f32 - (self.kernel_size.get() as f32) / 2.0;
                let y_f = y as f32 - (self.kernel_size.get() as f32) / 2.0;
                let cos = angle_rad.cos();
                let sin = angle_rad.sin();
                let g0_1 = sample_gaussian_2d_derivative_x(x_f, y_f, 1.0);
                let g90_1 = sample_gaussian_2d_derivative_y(x_f, y_f, 1.0);
                g0_1.mul_add(cos, g90_1 * sin)
              })
            })
            .collect::<Vec<f32>>(),
          self.kernel_size.get() as u32,
          self.kernel_size.get() as u32,
        )
        .unwrap()
      })
      .collect()
  }
}

/// What type of pyramid to compute. Each has different properties,
/// applications, and computation cost.
#[derive(Debug, Clone)]
pub enum ImagePyramidType {
  /// Use smoothing & subsampling to compute pyramid. This is used to generate
  /// mipmaps, thumbnails, display low-resolution previews of expensive image
  /// processing operations, texture synthesis, and more.
  Lowpass(SmoothingType),

  /// AKA Laplacian pyramid, where adjacent levels of the lowpass pyramid are
  /// upscaled and their pixel differences are computed. This used in image
  /// processing routines such as blending.
  Bandpass(SmoothingType),

  /// Uses a bank of multi-orientation bandpass filters. Used for used for
  /// applications including image compression, texture synthesis, and object
  /// recognition.
  Steerable(SteerableParams),
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
}

/// Generates a useful default set of parameters.
///
/// Defaults to a traditional image pyramid: Gaussian lowpass image pyramid with
/// scale factor of 0.5 and kernel size of 3.
impl Default for ImagePyramidParams {
  fn default() -> Self {
    Self {
      scale_factor: UnitIntervalValue::new(0.5).unwrap(),
      pyramid_type: ImagePyramidType::Lowpass(SmoothingType::Gaussian(OddValue(3))),
    }
  }
}

impl ImagePyramidParams {
  #[must_use]
  pub fn get_smoothing_type(&self) -> SmoothingType {
    match &self.pyramid_type {
      ImagePyramidType::Lowpass(smoothing_type) | ImagePyramidType::Bandpass(smoothing_type) =>
        smoothing_type.clone(),
      ImagePyramidType::Steerable(_) => SmoothingType::Gaussian(OddValue(3)),
    }
  }
}

/// The data associated with a given level of an image pyramid
pub enum ImagePyramidLevel {
  /// A single image representing the result from a filter.
  ///
  /// This is used for the lowpass and bandpass pyramids.
  Single(DynamicImage),

  /// A set of images representing the results from a filter bank.
  ///
  /// Depending on the filter bank used, the number of images may vary.
  ///
  /// In this crate, the steerable pyramid is the only filter bank implemented.
  /// Thus, the number of images will be equal to the number of orientations in
  /// the filter bank.
  Bank(Vec<DynamicImage>),
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
  /// Depending on the scale factor $s$, and image dimensions $(w, h)$,
  /// there will be $\lceil \log_{1/S}\min(w, h) \rceil$ levels.
  ///
  /// For example, an $(800, 600)$ image with scale factor $s=0.5$ will have
  /// $\lceil \log_{2}(600) \rceil=10$ levels.
  ///
  /// Similarly, a $(640, 480)$ image would have
  /// $\lceil \log_{2}(480) \rceil=9$ levels.
  pub levels: Vec<ImagePyramidLevel>,

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
      let smoothing_type = params.get_smoothing_type();
      let kernel = match smoothing_type {
        SmoothingType::Gaussian(k) => Kernel::<f32>::new_gaussian(&k)?,
        SmoothingType::Box(k) => Kernel::<f32>::new_box(&k)?,
        SmoothingType::Triangle(k) => Kernel::<f32>::new_triangle(&k)?,
      };
      let mut current_level = image.clone();
      #[allow(clippy::cast_possible_truncation)]
      #[allow(clippy::cast_precision_loss)]
      #[allow(clippy::cast_sign_loss)]
      while current_level.width() > 1 && current_level.height() > 1 {
        kernel.filter_in_place(&mut current_level, |c, a| *c = a as u8);
        current_level = current_level.resize_exact(
          (current_level.width() as f32 * params.scale_factor.get()) as u32,
          (current_level.height() as f32 * params.scale_factor.get()) as u32,
          image::imageops::FilterType::Gaussian,
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
      ImagePyramidType::Lowpass(_) =>
        Ok(ImagePyramid {
          levels: compute_lowpass_pyramid(self.0, &params)?
            .into_iter()
            .map(ImagePyramidLevel::Single)
            .collect(),
          params: params.clone(),
        }),
      ImagePyramidType::Bandpass(_) => {
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
          levels: levels.into_iter().map(ImagePyramidLevel::Single).collect(),
          params,
        })
      }
      ImagePyramidType::Steerable(steerable_params) => {
        let mut _current_level = self.0.clone();
        let mut _levels: Vec<ImagePyramidLevel> = Vec::new();
        let _kernels = steerable_params.get_kernels();
        Err(ImagePyramidError::NotImplemented(
          "Steerable pyramid".to_string(),
        ))
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use test_case::{test_case, test_matrix};

  use super::*;

  #[test_case(0.0, 1.0, 0.3989423)]
  #[test_case(1.0, 1.0, 0.24197073)]
  #[test_case(2.0, 1.0, 0.05399097)]
  fn sample_gaussian_1d_produces_expected_result(x: f32, sigma: f32, expected: f32) {
    let y = sample_gaussian_1d(x, sigma);
    assert_relative_eq!(y, expected, epsilon = 1e-6);
  }

  #[test]
  fn kernel_filter_in_place() {
    let mut image = DynamicImage::new_rgb8(3, 3);
    let mut other = DynamicImage::new_rgb8(3, 3);
    let mut i = 0;
    for y in 0..3 {
      for x in 0..3 {
        let mut pel = image.get_pixel(x, y);
        pel.apply_without_alpha(|_| i);
        image.put_pixel(x, y, pel);

        let mut pel = other.get_pixel(x, y);
        pel.apply_without_alpha(|_| i + 1);
        other.put_pixel(x, y, pel);
        i += 1;
      }
    }
    let kernel = Kernel::new_normalized(&[1u8, 2, 1, 2, 4, 2, 1, 2, 1], 3, 3).unwrap();
    kernel.filter_in_place(&mut image, |c, a| *c = a as u8);
    assert_eq!(image.get_pixel(1, 1), image::Rgba::<u8>([4, 4, 4, 255]));
  }

  #[test]
  fn kernel_new_error_when_dims_do_not_match_data() {
    let k = Kernel::new(&[1u8, 2, 3, 4, 5], 2, 2);
    assert!(k.is_err());
  }

  #[test]
  fn kernel_new_error_when_dims_are_zero() {
    let k = Kernel::new(&[1u8, 2, 3, 4, 5], 0, 0);
    assert!(k.is_err());
  }

  #[test]
  fn compute_image_pyramid_imagepyramidtype_steerable_unimplemented() {
    let image = DynamicImage::new_rgb8(640, 480);
    let ipr = ImageToProcess(&image);

    let params = ImagePyramidParams {
      pyramid_type: ImagePyramidType::Steerable(SteerableParams::default()),
      ..Default::default()
    };

    let pyramid = ipr.compute_image_pyramid(Some(&params));
    assert!(pyramid.is_err());
  }

  #[test_matrix(
    [
      ImagePyramidType::Lowpass(SmoothingType::Box(OddValue::new(3).unwrap())),
      ImagePyramidType::Lowpass(SmoothingType::Gaussian(OddValue::new(3).unwrap())),
      ImagePyramidType::Lowpass(SmoothingType::Triangle(OddValue::new(3).unwrap())),
      ImagePyramidType::Bandpass(SmoothingType::Box(OddValue::new(3).unwrap())),
      ImagePyramidType::Bandpass(SmoothingType::Gaussian(OddValue::new(3).unwrap())),
      ImagePyramidType::Bandpass(SmoothingType::Triangle(OddValue::new(3).unwrap()))
    ],
    [
      DynamicImage::new_luma16(128, 128),
      DynamicImage::new_luma8(128, 128),
      DynamicImage::new_luma_a16(128, 128),
      DynamicImage::new_luma_a8(128, 128),
      DynamicImage::new_rgb16(128, 128),
      DynamicImage::new_rgb8(128, 128),
      DynamicImage::new_rgb32f(128, 128),
      DynamicImage::new_rgba16(128, 128),
      DynamicImage::new_rgba8(128, 128),
      DynamicImage::new_rgba32f(128, 128)
    ]
  )]
  fn compute_image_pyramid(pyramid_type: ImagePyramidType, image: DynamicImage) {
    let ipr = ImageToProcess(&image);

    let params = ImagePyramidParams {
      pyramid_type: pyramid_type.clone(),
      ..Default::default()
    };

    let pyramid = ipr.compute_image_pyramid(Some(&params));
    assert!(pyramid.is_ok());
    let pyramid = pyramid.unwrap();
    assert_eq!(pyramid.levels.len(), 8);
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

  #[test]
  fn steerable_params_get_kernels() {
    let params = SteerableParams::new(4, 5.into_odd_value().unwrap());
    let kernels = params.get_kernels();
    assert_eq!(kernels.len(), 4);
    assert_eq!(kernels[0].width, 5);
    assert_eq!(kernels[0].height, 5);
  }
}
