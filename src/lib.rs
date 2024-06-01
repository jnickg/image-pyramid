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
//! - Steerable pyramids, which are explained under [`SteerableParams`] and
//!   [`ImagePyramidType::Steerable`]. This feature is disabled by default. See
//!   the [feature flags](#feature-flags) section for more information.
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
//! ## Feature Flags
//!
//! - `steerable` - Enables support for steerable pyramids. This feature is
//!   disabled by default, as it adds a dependency on the `nalgebra` and
//!   `ndarray` crates. If you need steerable pyramids, enable this feature in
//!   your `Cargo.toml` file.
//! - `rayon` - Enables parallel processing for pyramid computation. This is
//!   disabled by default, as it adds a dependency on the `rayon` crate. If you
//!   need parallel processing, enable this feature in your `Cargo.toml` file.
//!
//! ## Usage
//!
//! See the [crates.io page](https://crates.io/crates/image-pyramid) for
//! installation instructions, then check out the [examples directory](https://github.com/jnickg/image-pyramid/tree/main/examples)
//! for example code. Below is a simple illustrative example of computing a
//! default pyramid (Gaussian where each level is half resolution).
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
//! [`ImagePyramidParams::scale_factor`] field is a [`UnitIntervalValue`], which
//! must be a floating-point value in the interval (0, 1). Creating a value of
//! this type yields a [`Result`] and will contain an error if the value is not
//! valid.
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
//! Next, an example of creating a lowpass pyramid with a custom smoothing
//! kernel:
//!
//! ```rust
//! use image::DynamicImage;
//! use image_pyramid::*;
//!
//! let image = DynamicImage::new_rgba8(640, 480); // Or load from file
//!
//! // This happens to be a Gaussian kernel
//! let kernel = Kernel::new_normalized(
//!   &[1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0],
//!   kernel_size::THREE,
//!   kernel_size::THREE,
//! )
//! .unwrap();
//! let params = ImagePyramidParams {
//!   pyramid_type: ImagePyramidType::Lowpass(SmoothingType::CustomF32(kernel)),
//!   ..Default::default()
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
//! For any pyramid, the result is a [`ImagePyramid`] instance, which contains
//! the levels of the pyramid. Each level is a variant of [`ImagePyramidLevel`].
//! Unless a steerable pyramid is made, levels are always
//! [`ImagePyramidLevel::Single`], which contain a single `DynamicImage`
//! instance.
//!
//! ## Support & Contributing
//!
//! See the readme on the [GitHub page](http://www.github.com/jnickg/image-pyramid)

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
pub use std::num::NonZeroU8;

use image::{DynamicImage, GenericImage, GenericImageView, Pixel};
use num_traits::{clamp, Float, Num, NumCast};
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

  /// Raised when the user provides an invalid parameter somewhere that a more
  /// specific error is not available
  #[error("Bad parameter: {0}")]
  BadParameter(String),

  /// Raised when the requested functionality is not yet supported.
  #[error("Functionality \"{0}\" is not yet implemented.")]
  NotImplemented(String),

  /// Raised when the user requests a feature that is disabled at compile time.
  ///
  /// Features may be disabled at compile time due to large dependencies that
  /// users may want to exclude from their build, if they are not using the
  /// feature in question.
  #[error(
    "Feature \"{0}\" disabled at compile time. Rebuild with the appropriate feature flag. See \
     documentation for details."
  )]
  FeatureDisabled(String),

  /// Raised when something unexpected went wrong in the library.
  #[error("Internal error: {0}")]
  Internal(String),
}

/// A container for a value falling on the range $({0.0}--{1.0})$ (exclusive,
/// meaning the values $0.0$ and $1.0$ are not valid)
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

/// A simple 2D convolutional kernel that can be used to filter an image
///
/// The kernel is stored in row-major form, and the dimensions are provided
/// separately.
///
/// A kernel can be constructed with [`Kernel::new`], or
/// [`Kernel::new_normalized`]. These factories ensure the inputs are coherent
/// and valid, and return an error if they are not. These are most often used
/// with the [`SmoothingType::CustomF32`] variant, which allows a user-created
/// kernel to be used for smoothing.
///
/// Additional factories exist:
/// - [`Kernel::new_gaussian`] constructs a Gaussian kernel with the given size
/// - [`Kernel::new_triangle`] constructs a triangle kernel with the given size
/// - [`Kernel::new_box`] constructs a box kernel with the given size
///
/// However, these are more easily accessed by using the [`SmoothingType`] enum,
/// and are included mostly for completeness.
///
/// The kernel can be used to filter an image with the
/// [`Kernel::filter_in_place`] and [`Kernel::filter`] methods.
#[derive(Clone)]
pub struct Kernel<K: Num + Copy + Debug> {
  /// The elements of the kernel, stored in row-major layout. Its length is
  /// equal to $width \times height$
  pub data:   Vec<K>,
  /// The width of the kernel
  pub width:  u32,
  /// The height of the kernel
  pub height: u32,
}

impl<K: Num + Copy + Debug> Debug for Kernel<K> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "Kernel {{\n\twidth: {},\n\theight: {},\n\tdata: [",
      self.width, self.height
    )?;
    for (i, d) in self.data.iter().enumerate() {
      if i % self.width as usize == 0 {
        write!(f, "\n\t\t")?;
      }
      write!(f, "{d:?}\t")?;
    }
    write!(f, "\n\t],")?;
    write!(
      f,
      "\n\tsum: {:?}\n",
      self.data.iter().fold(K::zero(), |acc, x| acc + *x)
    )?;
    write!(f, "}}")
  }
}

/// An auxiliary trait used to determine in a value is sufficiently close to
/// zero that it likely was intended to be zero
///
/// This accounts for floating-point error when sampling continuous functions
/// like the Gaussian, where discretizing and normalizing may lead to
/// divide-by-zero errors. Instead, we check if the sum of the kernel is close
/// to zero with an epsilon.
pub trait CheckProbablyZero {
  fn is_probably_zero(&self) -> bool;
}

macro_rules! impl_check_if_zero_fp {
  ($($t:ty),*) => {
    $(
      impl CheckProbablyZero for $t {
        fn is_probably_zero(&self) -> bool {
          self.abs() < <$t>::epsilon()
        }
      }
    )*
  };
}
impl_check_if_zero_fp!(f32, f64);

macro_rules! impl_check_if_zero_int {
  ($($t:ty),*) => {
    $(
      impl CheckProbablyZero for $t {
        fn is_probably_zero(&self) -> bool {
          *self == 0
        }
      }
    )*
  };
}
impl_check_if_zero_int!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

/// An auxiliary trait describing values that can be used as elements in a
/// [`Kernel`]
pub trait KernelElement: Num + Copy + Debug + CheckProbablyZero + PartialOrd {}
macro_rules! impl_kernel_element {
  ($($t:ty),*) => {
    $(
      impl KernelElement for $t {}
    )*
  };
}
impl_kernel_element!(f32, f64, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

impl<K: KernelElement> Kernel<K> {
  /// Construct a kernel from a slice and its dimensions. The input slice is
  /// in row-major form. For example, a 3x3 matrix with data
  /// `[0,1,0,1,2,1,0,1,0`] describes the following matrix:
  ///
  /// $\begin{bmatrix} 0 & 1 & 0 \\\\ 1 & 2 & 1 \\\\ 0 & 1 & 0 \\\\
  /// \end{bmatrix}$
  ///
  /// # Errors
  ///
  /// - If the provided data does not match the size corresponding to the given
  ///   dimensions, [`ImagePyramidError::BadParameter`] is raised
  pub fn new(data: &[K], width: OddValue, height: OddValue) -> Result<Self, ImagePyramidError> {
    // Take the above asserts and return Internal error when appropriate
    let width = width.get() as u32;
    let height = height.get() as u32;
    if (width * height) as usize != data.len() {
      return Err(ImagePyramidError::BadParameter(format!(
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
  /// $\begin{bmatrix} 0 & 1 & 0 \\\\ 1 & 2 & 1 \\\\ 0 & 1 & 0 \\\\
  /// \end{bmatrix} / 6$
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
  pub fn new_normalized<K2>(
    data: &[K2],
    width: OddValue,
    height: OddValue,
  ) -> Result<Self, ImagePyramidError>
  where
    K: Float,
    K2: KernelElement + Into<K>,
  {
    let mut sum = K2::zero();
    let mut max = K2::zero();
    for i in data {
      sum = sum + *i;
      max = if *i > max { *i } else { max };
    }
    // If the sum is sufficiently close to zero, assume it is a derivative kernel,
    // meaning the DC component is supposed to zero. Normalize instead using the
    // maximum absolute value in the kernel. For floating point types, use an
    // epsilon for checking zero. For integet types, use strict equality
    let sum = if sum.is_probably_zero() { max } else { sum };

    let data_norm: Vec<K> = data
      .iter()
      .map(|x| <K2 as Into<K>>::into(*x) / <K2 as Into<K>>::into(sum))
      .collect();
    Self::new(&data_norm, width, height)
  }

  /// Creates a Gaussian kernel with the given kernel size.
  ///
  /// This is the most common kernel used for smoothing images, when computing
  /// an image pyramid. The kernel is normalized so that the sum of all elements
  /// is 1.0, which prevents distortion of the input signal.
  #[allow(clippy::missing_panics_doc)]
  #[must_use]
  pub fn new_gaussian(kernel_size: OddValue) -> Self
  where K: From<f32> + Into<f32> + Float {
    let k_size = kernel_size.get() as u32;
    let mut data = vec![K::zero(); (k_size * k_size) as usize];
    let sigma = 0.3f32.mul_add((k_size as f32 - 1.0).mul_add(0.5, -1.0), 0.8);
    for y in 0..k_size {
      for x in 0..k_size {
        let x_f = x as f32 - (k_size as f32) / 2.0;
        let y_f = y as f32 - (k_size as f32) / 2.0;
        let val = sample_gaussian_2d(x_f, y_f, sigma);
        let idx = (y * k_size + x) as usize;
        data[idx] = val.into();
      }
    }
    Self::new_normalized(&data, kernel_size, kernel_size).unwrap()
  }

  /// Creates a triangle kernel with the given kernel size.
  ///
  /// The triangle kernel is a simple kernel where a linear function is used to
  /// compute weights, where the max value is at $(0,0)$ (the center of the
  /// kernel).
  ///
  /// The results are normalized so that the kernel sums to 1.0, which prevents
  /// distortion of the input signal.
  #[allow(clippy::missing_panics_doc)]
  #[must_use]
  pub fn new_triangle(kernel_size: OddValue) -> Self
  where K: From<f32> + Into<f32> + Float {
    let k_size = kernel_size.get() as u32;
    let mut data = vec![K::zero(); (k_size * k_size) as usize];
    for y in 0..k_size {
      for x in 0..k_size {
        let x_f = x as f32 - (k_size as f32) / 2.0;
        let y_f = y as f32 - (k_size as f32) / 2.0;
        let val = sample_triangle_2d(x_f, y_f, 1.0);
        let idx = (y * k_size + x) as usize;
        data[idx] = val.into();
      }
    }
    Self::new_normalized(&data, kernel_size, kernel_size).unwrap()
  }

  /// Creates a box kernel with the given kernel size.
  ///
  /// The box kernel is a simple kernel where all elements are evenly weighted.
  /// The results are normalized so that the kernel sums to 1.0, which prevents
  /// distortion of the input signal.
  ///
  /// # Considerations
  ///
  /// It should be noted that box kernels are most often used because they can
  /// be implemented more cheaply than other kernels, such as Gaussian which
  /// usually requires use of floating-point math. Here, the box kernel also
  /// uses floating point math, so performance is likely equal to a Gaussian
  /// kernel.
  ///
  /// Instead, it is provided here for  compatibility with common image
  /// processing expectations, which may require use of a box kernel.
  #[allow(clippy::missing_panics_doc)]
  #[must_use]
  pub fn new_box(kernel_size: OddValue) -> Self
  where K: From<f32> + Into<f32> + Float {
    let k_size = kernel_size.get() as u32;
    let mut data = vec![K::zero(); (k_size * k_size) as usize];
    for y in 0..k_size {
      for x in 0..k_size {
        let val = 1.0;
        let idx = (y * k_size + x) as usize;
        data[idx] = val.into();
      }
    }
    Self::new_normalized(&data, kernel_size, kernel_size).unwrap()
  }

  /// Computes the 2D correlation of an image and this [`Kernel`] instance.
  ///
  /// Intermediate calculations are performed in a container of type $K$, and
  /// the results converted to pixel $Q$ via $f$. Pads image edges with
  /// continuity, meaning that the edge pixels are repeated to fill the
  /// kernel.
  ///
  /// # Safety
  ///
  /// This function is written to compute results quickly, so the inner loop
  /// contains an `unsafe` block that does not perform any bounds checking.
  /// Instead, the parameters of the loop are checked prior to entering the
  /// loop. This is safe because the loop parameters are derived from the image
  /// dimensions and kernel size.
  ///
  /// # Errors
  ///
  /// - If any of the image dimensions zero, [`ImagePyramidError::BadParameter`]
  ///   is raised.
  #[allow(unsafe_code)]
  #[allow(unused)]
  pub fn filter_in_place<I, F>(&self, image: &mut I, mut f: F) -> Result<(), ImagePyramidError>
  where
    I: GenericImage + Clone,
    <<I as GenericImageView>::Pixel as Pixel>::Subpixel: Into<K>,
    F: FnMut(&mut <<I as GenericImageView>::Pixel as Pixel>::Subpixel, K),
  {
    use core::cmp::{max, min};

    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
      return Err(ImagePyramidError::BadParameter(format!(
        "Image dimensions ({}, {}) are too small for kernel dimensions ({}, {})",
        width, height, self.width, self.height
      )));
    }

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
    Ok(())
  }

  /// Filters an image with this kernel, returning the result as a new image.
  ///
  /// See [`Kernel::filter_in_place`] for more information on filtering.
  ///
  /// # Errors
  /// - Whenever [`Kernel::filter_in_place`] yields an error.
  pub fn filter<I, F>(&self, image: &I, f: F) -> Result<I, ImagePyramidError>
  where
    I: GenericImage + Clone,
    <<I as GenericImageView>::Pixel as Pixel>::Subpixel: Into<K>,
    F: FnMut(&mut <<I as GenericImageView>::Pixel as Pixel>::Subpixel, K),
  {
    let mut image = image.clone();
    self.filter_in_place(&mut image, f)?;
    Ok(image)
  }
}

/// A simple wrapper extending the functionality of the given image with
/// image-pyramid support
pub struct ImageToProcess<'a>(pub &'a DynamicImage);

/// How to smooth an image when downsampling
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SmoothingType {
  /// Use a Gaussian kernel for smoothing, with the given kernel size.
  Gaussian(OddValue),
  /// Use a linear box filter for smoothing, with the given kernel size.
  Box(OddValue),
  /// Use a linear triangle filter for smoothing, with the given kernel size.
  Triangle(OddValue),
  /// Use a custom kernel for smoothing, defined by the user.
  ///
  /// User is responsible for ensuring the kernel produces coherent results.
  /// Generally, the kernel should have the following properties:
  ///
  /// - Normal ($\sum w_i = 1$): The sum of all elements in the kernel should be
  ///   1.0, to prevent distortion
  /// - Symmetric ($w_i = w_{-i}$): The kernel should be unbiased, meaning the
  ///   weights are symmetric about its center
  /// - Unimodal ($w_i \ge w_j \forall 0 < i < j$): The kernel should have a
  ///   single peak, meaning the weights are highest at the center and decrease
  ///   monotonically as distance from the center increases
  /// - Finite: The kernel should have a finite support, meaning the weights are
  ///   zero (or approach zero / discretize to zero) outside a certain radius
  ///   from the center. For example, a Gaussian kernel has infinite support,
  ///   but is often truncated to a finite size.
  /// - Size: The kernel should have an odd size, to ensure a well-defined
  ///   center
  CustomF32(Kernel<f32>),
}

/// A value that is guaranteed to be odd
///
/// This is useful for kernel sizes, which must be odd to have a well-defined
/// center
#[derive(Debug, Clone, Copy)]
pub struct OddValue(u8);

/// A set of constants for common kernel sizes. Most often
/// [`DEFAULT_KERNEL_SIZE`] is the best choice, but these are provided for ease
/// of use in the case that non-default values are needed.
pub mod kernel_size {
  use super::OddValue;

  pub const THREE: OddValue = OddValue(3);
  pub const FIVE: OddValue = OddValue(5);
  pub const SEVEN: OddValue = OddValue(7);
  pub const NINE: OddValue = OddValue(9);
  pub const ELEVEN: OddValue = OddValue(11);
}

/// The default kernel size. Users can pass this value wherever an [`OddValue`]
/// is required, if they don't want to specify a kernel size.
pub const DEFAULT_KERNEL_SIZE: OddValue = kernel_size::THREE;

/// The default kernel size for steerable pyramids. This is the most common
/// value used in practice.
pub const DEFAULT_STEERABLE_KERNEL_SIZE: OddValue = kernel_size::FIVE;

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

  /// Creates a new instance without checking if the value is odd
  ///
  /// # Safety
  /// - The value must be odd, as internal implementation in the `image-pyramid`
  ///   crate makes this assumption.
  #[allow(unsafe_code)]
  #[must_use]
  pub const unsafe fn new_unchecked(val: u8) -> Self { Self(val) }

  /// Retrieves the guaranteed odd value for use in computation.
  #[must_use]
  pub const fn get(self) -> u8 { self.0 }
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
/// This structure determines the number of orientations to compute, which are
/// evenly spaced around the unit circle. The kernel size determines the size of
/// the kernel to use for each filter.
///
/// # Background
///
/// - See [this page](http://www.cns.nyu.edu/~eero/steerpyr/) for a good
///   explanation of a steerable pyramid.
/// - [This page](https://flashypixels.wordpress.com/2017/02/17/derivative-of-gaussians-steerability/)
///   also has a great explanation of steerable filters in general, using the
///   Gaussian function.
///
/// Simply, put, a steerable pyramid is an image pyramid made up of multiple
/// bandpass subbands, each oriented in a different direction, and includes a
/// highpass subband for each non-terminal level, as well as the lowpass
/// residual at the terminal level.
///
/// _Steerable filters_ are a class of oriented filters that can be expressed as
/// a linear combination of a set of basis filters. As the most common example,
/// consider isotropic Gaussian filter $G(x,y)$:
/// - $G(x,y)=e^{−(x2+y2)}$
///
/// First derivative of $G(x,y)$ is given by $G_1$. Let $G^{\theta}_{1}$ be
/// the first derivative of $G$, rotated by an angle $\theta$ about the origin;
/// in the $x$ direction the angle is $\theta=0^{\circ}$ and in $y$ direction,
/// $\theta=90^{\circ}$. The first derivatives in $x$ and $y$ directions are
/// given by:
/// - $G^{0^{\circ}}_1 = {{\partial{G}}\over{\partial{x}}} = −2xe^{−(x^2+y^2)}$,
///   and
/// - $G^{90^{\circ}}_1 = {{\partial{G}}\over{\partial{y}}} = −2ye^{−(x^2+y^2)}$
///
/// In 2D space $G^{0^{\circ}}_1$ and $G^{90^{\circ}}_1$ are seen to span the
/// entire space and are the basis filters. Thus, an arbitrarily-oriented first
/// derivative filter can be expressed as a linear combination of these two
/// filters:
///
/// $G^{θ^{\circ}}_1 = G^{0^{\circ}}_1 \cos{\theta} + G^{90^{\circ}}_1
/// \sin{\theta}$
///
/// This is how one bandpass subband filter is computed for a steerable pyramid.
/// For a steerable pyramid with $n$ orientations, $n$ bandpass subbands are
/// computed, each rotated by an angle of $360 \over n$ degrees.
///
/// # Example
///
/// This example creates a steerable pyramid with four orientations and a kernel
/// size of 5. A kernel size of three is not recommended for steerable pyramids,
/// as the filters may not well-defined with such a small kernel.
///
/// ```rust
/// use image::DynamicImage;
/// use image_pyramid::*;
///
/// let image = DynamicImage::new_rgba8(640, 480); // Or load from file
///
/// // Four directional filters, with a kernel size of 5, the canonical steerable pyramid.
/// let steerable_params =
///   SteerableParams::new(NonZeroU8::new(4).unwrap(), OddValue::new(5).unwrap());
/// let params = ImagePyramidParams {
///   pyramid_type: ImagePyramidType::Steerable(steerable_params),
///   ..Default::default()
/// };
/// let pyramid = match ImagePyramid::create(&image, Some(&params)) {
///   Ok(pyramid) => pyramid,
///   Err(e) => {
///     eprintln!("Error creating image pyramid: {}", e);
///     return;
///   }
/// };
/// ```
///
/// # Implementation
///
/// This crate takes advantage of the fact that $ f^{\theta} * I = \sum_{i}
/// k_i{(\theta)}(f^i*I) $, where
/// - $ f^{\theta} $ is the steerable filter at angle $ \theta $,
/// - $ f^i $ is the $ i $th basis filter,
/// - $ I $ is the input image,
/// - $ k_i{(\theta)} $ is the steering function (weight) for $ f_i $ at angle $
///   \theta $.
///
/// That is, for each level of the pyramid, only two convolutions are performed
/// for the basis functions. Then, each steerable subband is computed by a
/// linear combination of the results of the basis functions.
#[derive(Debug, Clone, Copy)]
pub struct SteerableParams {
  /// The number of orientations to compute.
  ///
  /// This is the number of filters to
  /// compute. See [`SteerableParams`] for explanation.
  ///
  /// The default value is defined by [`DEFAULT_STEERABLE_NUM_ORIENTATIONS`].
  pub num_orientations: NonZeroU8,

  /// The size of the kernel to use for each filter. This is the size of the
  /// kernel in both the $x$ and $y$ directions.
  ///
  /// The default value is defined by [`DEFAULT_STEERABLE_KERNEL_SIZE`].
  pub kernel_size: OddValue,
}

/// The four cardinal orientations (0, 90, 180, 270 degrees).
#[allow(unsafe_code)]
pub const DEFAULT_STEERABLE_NUM_ORIENTATIONS: NonZeroU8 = unsafe { NonZeroU8::new_unchecked(4) };

/// The default set of parameters for computing steerable pyramids.
///
/// This is the most common set of parameters used in practice, and thus a
/// useful default for users who don't need to customize the parameters.
impl Default for SteerableParams {
  fn default() -> Self {
    Self {
      num_orientations: DEFAULT_STEERABLE_NUM_ORIENTATIONS,
      kernel_size:      DEFAULT_STEERABLE_KERNEL_SIZE,
    }
  }
}

pub(crate) fn sample_triangle_1d(x: f32, w: f32) -> f32 { 1.0 - (x / w).abs() }

pub(crate) fn sample_triangle_2d(x: f32, y: f32, w: f32) -> f32 {
  sample_triangle_1d(x, w) * sample_triangle_1d(y, w)
}

/// Samples the Gaussian function at the given point $x$ with standard deviation
/// $\sigma$
pub(crate) fn sample_gaussian_1d(x: f32, sigma: f32) -> f32 {
  ((2.0 * std::f32::consts::PI).sqrt() * sigma).recip() * (-x.powi(2) / (2.0 * sigma.powi(2))).exp()
}

pub(crate) fn sample_gaussian_2d(x: f32, y: f32, sigma: f32) -> f32 {
  sample_gaussian_1d(x, sigma) * sample_gaussian_1d(y, sigma)
}

/// Computes the gradient of the given 2D vector with respect to X
///
/// The gradient is computed using second order accurate central differences in
/// the interior points and either first or second order accurate one-sides
/// (forward or backwards) differences at the boundaries. The returned gradient
/// hence has the same shape as the input array.
///
/// The gradient with respect to x is computed as $a(x+1, y) - a(x-1, y)$
pub(crate) fn gradient_x<T: Num + Copy>(a: &Vec<T>, w: usize, h: usize) -> Vec<T> {
  assert_eq!(
    a.len(),
    w * h,
    "Input array must have the same shape as the image"
  );
  let mut grad = vec![T::zero(); w * h];
  for y in 0..h {
    for x in 0..w {
      let x = x as isize;
      let y = y as isize;
      let idx = y * w as isize + x;
      let idx = idx as usize;
      let x = x as usize;
      let y = y as usize;
      let x0 = if x > 0 { x - 1 } else { x };
      let x1 = if x < w - 1 { x + 1 } else { x };
      let idx0 = y * w + x0;
      let idx1 = y * w + x1;
      grad[idx] = a[idx1] - a[idx0];
    }
  }
  grad
}

/// Computes the gradient of the given 2D vector with respect to Y
///
/// See [`gradient_x`] for more general information on gradients
pub(crate) fn gradient_y<T: Num + Copy>(a: &Vec<T>, w: usize, h: usize) -> Vec<T> {
  assert_eq!(
    a.len(),
    w * h,
    "Input array must have the same shape as the image"
  );
  let mut grad = vec![T::zero(); w * h];
  for y in 0..h {
    for x in 0..w {
      let x = x as isize;
      let y = y as isize;
      let idx = y * w as isize + x;
      let idx = idx as usize;
      let x = x as usize;
      let y = y as usize;
      let y0 = if y > 0 { y - 1 } else { y };
      let y1 = if y < h - 1 { y + 1 } else { y };
      let idx0 = y0 * w + x;
      let idx1 = y1 * w + x;
      grad[idx] = a[idx1] - a[idx0];
    }
  }
  grad
}

pub(crate) fn _factorial(number: i128) -> i128 {
  let mut factorial: i128 = 1;
  for i in 1..=number {
    factorial *= i;
  }
  factorial
}

impl SteerableParams {
  /// Creates a new instance of [`SteerableParams`] with the given values.
  ///
  /// No additional checking is used beyond what the types themselves enforce.
  #[must_use]
  pub const fn new(num_orientations: NonZeroU8, kernel_size: OddValue) -> Self {
    Self {
      num_orientations,
      kernel_size,
    }
  }

  /// Creates a new instance of [`SteerableParams`] with the given values.
  ///
  /// This is a convenience method that checks the values to ensure they are
  /// valid. If they are not, an error is returned.
  ///
  /// Consider using [`SteerableParams::new`] for type-level protection of
  /// values.
  ///
  /// # Errors
  ///
  /// - If `num_orientations` is zero
  /// - If `kernel_size` is not odd
  pub fn from_primitives(num_orientations: u8, kernel_size: u8) -> Result<Self, ImagePyramidError> {
    Ok(Self {
      num_orientations: NonZeroU8::new(num_orientations).ok_or_else(|| {
        ImagePyramidError::BadParameter("num_orientations must be non-zero".to_string())
      })?,
      kernel_size:      OddValue::new(kernel_size)?,
    })
  }

  /// Computes the basis kernels for the given sigma value.
  ///
  /// The basis kernels are the first-order derivatives of the Gaussian kernel
  /// with respect to $x$ and $y$. These are the basis functions for the
  /// steerable pyramid.
  ///
  /// # Errors
  /// Any errors are due to internal errors in creating the kernels, and are
  /// a [`ImagePyramidError::Internal`] error. If this occurs, it is likely due
  /// to a bug in the implementation. Please report this error [here](https://github.com/jnickg/image-pyramid/issues).
  pub fn get_basis_kernels_with_sigma(
    &self,
    sigma: f32,
  ) -> Result<(Kernel<f32>, Kernel<f32>), ImagePyramidError> {
    let kernel_size = self.kernel_size.get() as usize;
    let k_size_f32 = self.kernel_size.get() as f32;
    let k_idx_offset = k_size_f32 / 2.0 - 0.5;
    let order = 1;

    let mut gaussian = Vec::with_capacity(kernel_size * kernel_size);
    for y in 0..kernel_size as i32 {
      for x in 0..kernel_size as i32 {
        let x_f = x as f32 - k_idx_offset;
        let y_f = y as f32 - k_idx_offset;
        let val = sample_gaussian_2d(x_f, y_f, sigma);
        gaussian.push(val);
      }
    }
    let gaussian =
      Kernel::<f32>::new_normalized(&gaussian, self.kernel_size, self.kernel_size).unwrap();
    dbg!(&gaussian);

    let mut basis_x = gaussian.data.clone();
    let mut basis_y = gaussian.data.clone();

    // Compute the partial derivative of the Gaussian kernel with respect to x and y
    for _ in 0..order {
      basis_x = gradient_x(&basis_x, kernel_size, kernel_size);
      basis_y = gradient_y(&basis_y, kernel_size, kernel_size);
    }

    dbg!(&basis_x);
    dbg!(&basis_y);

    let sum_basis_x: f32 = basis_x.iter().sum();
    let sum_basis_y: f32 = basis_y.iter().sum();
    dbg!((sum_basis_x, sum_basis_y));

    let basis_x = Kernel::<f32>::new_normalized(&basis_x, self.kernel_size, self.kernel_size)
      .map_err(|e| ImagePyramidError::Internal(format!("Error creating kernel: {e}")))?;

    let basis_y = Kernel::<f32>::new_normalized(&basis_y, self.kernel_size, self.kernel_size)
      .map_err(|e| ImagePyramidError::Internal(format!("Error creating kernel: {e}")))?;

    dbg!(&basis_x);
    dbg!(&basis_y);

    Ok((basis_x, basis_y))
  }

  /// Determines an appropriate $\sigma$ (standard deviation) value for the
  /// given kernel size.
  ///
  /// As a convention, this is one third of the kernel size.
  #[must_use]
  pub fn sigma(&self) -> f32 { self.kernel_size.get() as f32 / 3.0 }

  /// Gets the angles for each directional filter in the steerable pyramid.
  ///
  /// These are automatically computed based on
  /// [`SteerableParams::num_orientations`].
  #[must_use]
  pub fn get_angles(&self) -> Vec<f32> {
    (0..self.num_orientations.get())
      .map(|i| std::f32::consts::PI * i as f32 / self.num_orientations.get() as f32)
      .collect()
  }

  /// Computes the basis kernels with an automatic value for $\sigma$.
  ///
  /// # Errors
  /// See [`SteerableParams::get_basis_kernels_with_sigma`] for more
  /// information.
  pub fn get_basis_kernels(&self) -> Result<(Kernel<f32>, Kernel<f32>), ImagePyramidError> {
    self.get_basis_kernels_with_sigma(self.sigma())
  }

  /// Computes the angles for each filter in the steerable pyramid.
  ///
  /// # Errors
  /// Errors are due to internal errors in computing the angles, and are a
  /// [`ImagePyramidError::Internal`] error. If this occurs, it is likely due to
  /// a bug in the implementation. Please report this error [here](https://github.com/jnickg/image-pyramid/issues).
  pub fn get_kernels_with_sigma(&self, sigma: f32) -> Result<Vec<Kernel<f32>>, ImagePyramidError> {
    // First, get the basis functions
    let (basis_x, basis_y) = self.get_basis_kernels_with_sigma(sigma)?;

    // Then, steer the basis functions to get the steerable filters
    let angles = self.get_angles();

    let kernels = angles
      .iter()
      .map(|theta| {
        let cos_theta = theta.cos(); // The steering function for the x basis filter
        let sin_theta = theta.sin(); // The steering function for the y basis filter

        // $G_{\theta} = G_x \cos{\theta} + G_y \sin{\theta}$
        let data: Vec<f32> = basis_x
          .data
          .iter()
          .zip(basis_y.data.iter())
          .map(|(bx, by)| bx * cos_theta + by * sin_theta)
          .collect();
        let steered_kernel = Kernel::new(&data, self.kernel_size, self.kernel_size);
        steered_kernel
      })
      .collect::<Result<Vec<Kernel<f32>>, ImagePyramidError>>()?;

    Ok(kernels)
  }

  /// This takes the number of orientations and computes the first-order
  /// derivative-of-Gaussian kernels for each orientation.
  ///
  /// It starts by computing the angles for each filter, then creates the two
  /// basis filters, and then computes each steerable filter as a linear
  /// combination of the two basis filters.
  ///
  /// # Errors
  /// See [`SteerableParams::get_kernels_with_sigma`] for more information.
  pub fn get_kernels(&self) -> Result<Vec<Kernel<f32>>, ImagePyramidError> {
    self.get_kernels_with_sigma(self.sigma())
  }
}

/// What type of pyramid to compute. Each has different properties,
/// applications, and computation cost.
#[derive(Debug, Clone)]
#[non_exhaustive]
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
  /// recognition. See [`SteerableParams`] for more information.
  ///
  /// When this type is used, the result produces a [`ImagePyramidLevel::Bank`]
  /// result
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
      pyramid_type: ImagePyramidType::Lowpass(SmoothingType::Gaussian(kernel_size::THREE)),
    }
  }
}

impl ImagePyramidParams {
  /// Gets the [`SmoothingType`] to use when computing a lowpass pyramid with
  /// the current parameters.
  ///
  /// # Notes
  /// - If the pyramid type is [`ImagePyramidType::Steerable`], this will return
  ///   a Gaussian kernel with a kernel size of 3. While this is what is used to
  ///   compute the lowpass portion of a steerable pyramid, it does _not_
  ///   describe the Derivative of Gaussian kernels, which are dynamically
  ///   computed, used when computing the bandpass portion of a steerable
  ///   pyramid
  #[must_use]
  pub fn get_lowpass_smoothing_type(&self) -> SmoothingType {
    match &self.pyramid_type {
      ImagePyramidType::Lowpass(smoothing_type) | ImagePyramidType::Bandpass(smoothing_type) =>
        smoothing_type.clone(),
      ImagePyramidType::Steerable(_) => SmoothingType::Gaussian(kernel_size::THREE),
    }
  }
}

/// The data associated with a given level of an image pyramid
#[derive(Debug)]
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
#[derive(Debug)]
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
  /// - Errors of type [`ImagePyramidError`] are raised if the image pyramid
  ///   cannot be computed for some reason.
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
      let smoothing_type = params.get_lowpass_smoothing_type();
      let kernel = match smoothing_type {
        SmoothingType::Gaussian(k_size) => Kernel::<f32>::new_gaussian(k_size),
        SmoothingType::Box(k_size) => Kernel::<f32>::new_box(k_size),
        SmoothingType::Triangle(k_size) => Kernel::<f32>::new_triangle(k_size),
        SmoothingType::CustomF32(k) => k,
      };
      let mut current_level = image.clone();
      #[allow(clippy::cast_possible_truncation)]
      #[allow(clippy::cast_precision_loss)]
      #[allow(clippy::cast_sign_loss)]
      while current_level.width() > 1 && current_level.height() > 1 {
        kernel.filter_in_place(&mut current_level, |c, a| *c = a as u8)?;
        current_level = current_level.resize_exact(
          (current_level.width() as f32 * params.scale_factor.get()) as u32,
          (current_level.height() as f32 * params.scale_factor.get()) as u32,
          image::imageops::FilterType::Nearest,
        );
        levels.push(current_level.clone());
      }
      Ok(levels)
    }

    // fn lowpass_in_place(image: &mut DynamicImage, params: &ImagePyramidParams) ->
    // Result<(), ImagePyramidError> {   let smoothing_type =
    // params.get_lowpass_smoothing_type();   let kernel = match smoothing_type
    // {     SmoothingType::Gaussian(k_size) =>
    // Kernel::<f32>::new_gaussian(k_size),     SmoothingType::Box(k_size) =>
    // Kernel::<f32>::new_box(k_size),     SmoothingType::Triangle(k_size) =>
    // Kernel::<f32>::new_triangle(k_size),     SmoothingType::CustomF32(k) =>
    // k,   };
    //   kernel.filter_in_place(image, |c, a| *c = a as u8)
    // }

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

    fn compute_bandpass_pyramid(
      image: &DynamicImage,
      params: &ImagePyramidParams,
    ) -> Result<Vec<DynamicImage>, ImagePyramidError> {
      // First, we need a lowpass pyramid to work with.
      let mut levels = compute_lowpass_pyramid(image, &params)?;

      // For each index $i$, upscale the resolution of pyramid level $L_{i+1}$ to
      // match the resolution of pyramid level $L_i$. Then we compute the pixel-wise
      // difference between them, and store the result in the current level.
      for i in 0..levels.len() - 1 {
        let next_level = levels[i + 1].resize_exact(
          levels[i].width(),
          levels[i].height(),
          image::imageops::FilterType::Nearest,
        );
        bandpass_in_place(&mut levels[i], &next_level);
      }

      Ok(levels)
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
      ImagePyramidType::Bandpass(_) =>
        Ok(ImagePyramid {
          levels: compute_bandpass_pyramid(self.0, &params)?
            .into_iter()
            .map(ImagePyramidLevel::Single)
            .collect(),
          params: params.clone(),
        }),
      #[cfg(not(feature = "steerable"))]
      ImagePyramidType::Steerable(_) =>
        Err(ImagePyramidError::FeatureDisabled("steerable".to_string())),
      #[cfg(feature = "steerable")]
      ImagePyramidType::Steerable(steerable_params) => {
        // At each level, we compute:
        // - The bandpass (high pass) subband H_n
        // - The lowpass (residual) subband L_n
        // - The steerable subbands which are the result of convolving L_n with the
        //   basis filters, then for each orientation, steering the filter response to
        //   the desired orientation.
        //
        // Then, we subsample L_n to get L_{n+1}, and repeat the process.

        // First, compute the lowpass pyramid
        #[allow(nonstandard_style)]
        let L = compute_lowpass_pyramid(self.0, &params)?;
        #[allow(nonstandard_style)]
        let H = compute_bandpass_pyramid(self.0, &params)?;

        // Compute the steerable pyramid
        let (basis_kernel_x, basis_kernel_y) = steerable_params.get_basis_kernels()?;

        // Convolutions are done in-place, so we need to clone the lowpass pyramid
        // for each of the basis kernels
        let mut basis_pyramid_x = L.clone();
        let mut basis_pyramid_y = L.clone();
        basis_pyramid_x.iter_mut().for_each(|l| {
          basis_kernel_x
            .filter_in_place(l, |c, a| *c = a as u8)
            .unwrap()
        });
        basis_pyramid_y.iter_mut().for_each(|l| {
          basis_kernel_y
            .filter_in_place(l, |c, a| *c = a as u8)
            .unwrap()
        });

        // Now that we have the basis responses for each pyramid level, we can
        // steer those responses for each of the angles to form our levels
        let angles = steerable_params.get_angles();

        // Levels is a vector of vectors, where each inner vector is the steered
        // response for one of the angles, and the outer vector is the set of
        // all steered responses arranged by pyramid level.
        let mut levels = Vec::with_capacity(L.len());

        for (l, (l_bx, l_by)) in L
          .iter()
          .zip(basis_pyramid_x.iter().zip(basis_pyramid_y.iter()))
        {
          let mut level = Vec::with_capacity(angles.len());
          for angle in angles.iter() {
            let cos_theta = angle.cos();
            let sin_theta = angle.sin();
            let mut steered = l.clone();
            // Iterate through the mutable pixels in `steered`, and for each one set it to
            // the value of `l_bx_pixel as f32 * cos_theta + l_by_pixel as f32 *
            // sin_theta`, where `l_bx_pixel` and `l_by_pixel` are the
            // corresponding pixels in the basis pyramids.
            for y in 0..steered.height() {
              for x in 0..steered.width() {
                let l_bx_pixel = l_bx.get_pixel(x, y);
                let l_by_pixel = l_by.get_pixel(x, y);
                let mut pixel = steered.get_pixel(x, y);
                pixel
                  .channels_mut()
                  .into_iter()
                  .enumerate()
                  .for_each(|(i, c)| {
                    // need to rescale 0-255 to 0-1 first, then scale back to 0-255
                    let l_bx_val = l_bx_pixel[i] as f32 / 255.0;
                    let l_by_val = l_by_pixel[i] as f32 / 255.0;
                    let result = l_bx_val * cos_theta + l_by_val * sin_theta;
                    *c = (result * 255.0) as u8;
                    // *c = (l_bx_val * cos_theta + l_by_val * sin_theta) as u8;
                  });
                steered.put_pixel(x, y, pixel);
              }
            }
            level.push(steered);
          }
          levels.push(level);
        }

        // Now, append to each level the corresponding level in H
        for (l, h) in levels.iter_mut().zip(H.iter()) {
          l.push(h.clone());
        }

        Ok(ImagePyramid {
          levels: levels.into_iter().map(ImagePyramidLevel::Bank).collect(),
          params: params.clone(),
        })
      }
    }
  }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
mod tests {
  use test_case::{test_case, test_matrix};

  use super::{kernel_size::*, *};

  #[test_case(0, 1)]
  #[test_case(1, 1)]
  #[test_case(2, 2)]
  #[test_case(3, 6)]
  #[test_case(4, 24)]
  #[test_case(5, 120)]
  #[test_case(6, 720)]
  #[test_case(7, 5040)]
  #[test_case(8, 40320)]
  #[test_case(9, 362880)]
  #[test_case(10, 3628800)]
  fn factorial_produces_expected_results(input: i128, expected: i128) {
    assert_eq!(_factorial(input), expected);
  }

  #[test]
  fn gradient_x_produces_expected_results() {
    let a = vec![1, 2, 1, 2, 4, 2, 1, 2, 1];
    let grad = gradient_x(&a, 3, 3);
    let expected = vec![1, 0, -1, 2, 0, -2, 1, 0, -1];
    assert_eq!(grad, expected);
  }

  #[test]
  fn gradient_x_f32_produces_expected_results() {
    #[rustfmt::skip]
    let a = vec![
      0.07511362, 0.12384141, 0.07511362,
      0.12384141, 0.20417997, 0.12384141,
      0.07511362, 0.12384141, 0.07511362,
    ];
    assert_relative_eq!(1.0, a.iter().sum::<f32>());
    let grad = gradient_x(&a, 3, 3);
    #[rustfmt::skip]
    let expected = vec![
      0.048727795, 0.0, -0.048727795,
      0.08033856,  0.0, -0.08033856,
      0.048727795, 0.0, -0.048727795,
    ];
    assert_eq!(grad, expected);
  }

  #[test]
  fn gradient_y_produces_expected_results() {
    let a = vec![1, 2, 1, 2, 4, 2, 1, 2, 1];
    let expected = vec![1, 2, 1, 0, 0, 0, -1, -2, -1];
    let grad = gradient_y(&a, 3, 3);
    assert_eq!(grad, expected);
  }

  #[test]
  fn gradient_y_f32_produces_expected_results() {
    #[rustfmt::skip]
    let a = vec![
      0.07511362, 0.12384141, 0.07511362,
      0.12384141, 0.20417997, 0.12384141,
      0.07511362, 0.12384141, 0.07511362,
    ];
    assert_relative_eq!(1.0, a.iter().sum::<f32>());
    let grad = gradient_y(&a, 3, 3);
    #[rustfmt::skip]
    let expected = vec![
      0.048727795,   0.08033856,  0.048727795,
      0.0,           0.0,         0.0,
      -0.048727795, -0.08033856, -0.048727795,
    ];
    assert_eq!(grad, expected);
  }

  #[test_case(0.0, 1.0, 0.398_942_3)]
  #[test_case(1.0, 1.0, 0.241_970_73)]
  #[test_case(2.0, 1.0, 0.053_990_97)]
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
    let kernel =
      Kernel::<f32>::new_normalized(&[1u8, 2, 1, 2, 4, 2, 1, 2, 1], THREE, THREE).unwrap();
    let result = kernel.filter_in_place(&mut image, |c, a| *c = a as u8);
    assert!(result.is_ok());
    assert_eq!(image.get_pixel(1, 1), image::Rgba::<u8>([4, 4, 4, 255]));
  }

  #[test]
  fn kernel_new_error_when_dims_do_not_match_data() {
    let k = Kernel::new(&[1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10], THREE, THREE);
    assert!(k.is_err());
  }

  #[test_matrix([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])]
  fn kernel_new_gaussian_never_fails(kernel_size: u8) {
    let k_size = OddValue::new(kernel_size);
    assert!(k_size.is_ok());
    let _ = Kernel::<f32>::new_gaussian(k_size.unwrap());
  }

  #[test_matrix([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])]
  fn kernel_new_triangle_never_fails(kernel_size: u8) {
    let k_size = OddValue::new(kernel_size);
    assert!(k_size.is_ok());
    let _ = Kernel::<f32>::new_triangle(k_size.unwrap());
  }

  #[test_matrix([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])]
  fn kernel_new_box_never_fails(kernel_size: u8) {
    let k_size = OddValue::new(kernel_size);
    assert!(k_size.is_ok());
    let _ = Kernel::<f32>::new_box(k_size.unwrap());
  }

  #[test]
  #[cfg_attr(not(feature = "steerable"), ignore)]
  fn compute_image_pyramid_imagepyramidtype_steerable_produces_expected_filter_bank() {
    let image = DynamicImage::new_rgb8(128, 128);
    let ipr = ImageToProcess(&image);

    let params = ImagePyramidParams {
      pyramid_type: ImagePyramidType::Steerable(SteerableParams::default()),
      ..Default::default()
    };

    let pyramid = ipr.compute_image_pyramid(Some(&params));
    assert!(pyramid.is_ok());
    let pyramid = pyramid.unwrap();
    assert_eq!(pyramid.levels.len(), 8);
    // Each level should have the correct default number of steered responses, plus
    // a highpass response for each level
    for level in pyramid.levels.iter() {
      if let ImagePyramidLevel::Bank(bank) = level {
        assert_eq!(bank.len(), 5);
      } else {
        panic!("Expected ImagePyramidLevel::Bank, got something else");
      }
    }
  }

  #[test]
  #[cfg_attr(feature = "steerable", ignore)]
  fn compute_image_pyramid_imagepyramidtype_steerable_disabled() {
    let image = DynamicImage::new_rgb8(640, 480);
    let ipr = ImageToProcess(&image);

    let params = ImagePyramidParams {
      pyramid_type: ImagePyramidType::Steerable(SteerableParams::default()),
      ..Default::default()
    };

    let pyramid = ipr.compute_image_pyramid(Some(&params));
    assert!(pyramid.is_err());
    // And the error should be that the feature is disabled
    let err = pyramid.unwrap_err();
    assert!(matches!(err, ImagePyramidError::FeatureDisabled(_)));
  }

  #[test_matrix(
    [
      &ImagePyramidType::Lowpass(SmoothingType::Box(THREE)),
      &ImagePyramidType::Lowpass(SmoothingType::Gaussian(THREE)),
      &ImagePyramidType::Lowpass(SmoothingType::Triangle(THREE)),
      &ImagePyramidType::Bandpass(SmoothingType::Box(THREE)),
      &ImagePyramidType::Bandpass(SmoothingType::Gaussian(THREE)),
      &ImagePyramidType::Bandpass(SmoothingType::Triangle(THREE))
    ],
    [
      &DynamicImage::new_luma16(128, 128),
      &DynamicImage::new_luma8(128, 128),
      &DynamicImage::new_luma_a16(128, 128),
      &DynamicImage::new_luma_a8(128, 128),
      &DynamicImage::new_rgb16(128, 128),
      &DynamicImage::new_rgb8(128, 128),
      &DynamicImage::new_rgb32f(128, 128),
      &DynamicImage::new_rgba16(128, 128),
      &DynamicImage::new_rgba8(128, 128),
      &DynamicImage::new_rgba32f(128, 128)
    ]
  )]
  fn compute_image_pyramid(pyramid_type: &ImagePyramidType, image: &DynamicImage) {
    let ipr = ImageToProcess(image);

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
    let i = 0.5.into_unit_interval();
    assert!(i.is_ok());
    assert_relative_eq!(0.5, i.unwrap().get());
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
