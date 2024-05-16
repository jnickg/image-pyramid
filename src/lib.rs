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

//! # Image Pyramid Crate
//!
//! TODO copy from README.md

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

use image::DynamicImage;

pub struct ImageToProcess<'a>(pub &'a DynamicImage);

/// How to smooth an image when downsampling
pub enum SmoothingType {
  /// Use a Gaussian filter
  /// `[[1,2,1],[2,4,2],[1,2,1]] * 1/16`
  Gaussian,
  /// Use a linear box filter
  /// `[[1,1,1],[1,1,1],[1,1,1]] * 1/9`
  Box,
  /// Use a linear triangle filter:
  /// `[[1,2,1],[2,4,2],[1,2,1]] * 1/16`
  Triangle
}

impl Clone for SmoothingType {
  fn clone(&self) -> Self {
    match self {
      SmoothingType::Gaussian => SmoothingType::Gaussian,
      SmoothingType::Box => SmoothingType::Box,
      SmoothingType::Triangle => SmoothingType::Triangle
    }
  }
}

/// What type of pyramid to compute. Each has different properties, applications, and computation
/// cost.
pub enum ImagePyramidType {
  /// Use smoothing & subsampling to compute pyramid. This is used to generate mipmaps, thumbnails,
  /// display low-resolution previews of expensive image processing operations, texture
  /// synthesis, and more.
  Lowpass,

  /// AKA Laplacian pyramid, where adjacent levels of the lowpass pyramid are upscaled and their
  /// pixel differences are computed. This used in image processing routines such as blending.
  Bandpass,

  /// Uses a bank of multi-orientation bandpass filters. Used for used for applications including
  /// image compression, texture synthesis, and object recognition.
  Steerable
}

impl Clone for ImagePyramidType {
  fn clone(&self) -> Self {
    match self {
      ImagePyramidType::Lowpass => ImagePyramidType::Lowpass,
      ImagePyramidType::Bandpass => ImagePyramidType::Bandpass,
      ImagePyramidType::Steerable => ImagePyramidType::Steerable
    }
  }
}

pub struct ImagePyramidParams {
  /// The scale factor to use on image dimensions when downsampling. This is most commonly 0.5
  scale_factor: f32,
  ///
  pyramid_type: ImagePyramidType,
  smoothing_type: SmoothingType
}

impl Default for ImagePyramidParams {
  fn default() -> Self {
    ImagePyramidParams {
      scale_factor: 0.5,
      pyramid_type: ImagePyramidType::Lowpass,
      smoothing_type: SmoothingType::Gaussian
    }
  }
}

impl Clone for ImagePyramidParams {
  fn clone(&self) -> Self {
    ImagePyramidParams {
      scale_factor: self.scale_factor,
      pyramid_type: self.pyramid_type.clone(),
      smoothing_type: self.smoothing_type.clone()
    }
  }
}

pub struct ImagePyramid {
  pub levels: Vec<DynamicImage>,
  pub params: ImagePyramidParams
}

impl ImagePyramid {
  pub fn new () -> Self {
    ImagePyramid {
      levels: Vec::new(),
      params: ImagePyramidParams::default()
    }
  }

  pub fn with(params: &ImagePyramidParams) -> Self {
    ImagePyramid {
      levels: Vec::new(),
      params: params.clone()
    }
  }

  pub fn create(image: &DynamicImage, params: &ImagePyramidParams) -> Result<Self, &'static str> {
    let image_to_process = ImageToProcess(image);
    let pyramid = image_to_process.compute_image_pyramid(&params)?;

    Ok(pyramid)
  }
}

pub trait CanComputePyramid {
  fn compute_image_pyramid(
    &self,
    params: &ImagePyramidParams
  ) -> Result<ImagePyramid, &'static str>;
}

impl<'a> CanComputePyramid for ImageToProcess<'a> {
  fn compute_image_pyramid(
    &self,
    params: &ImagePyramidParams
  ) -> Result<ImagePyramid, &'static str> {
    match params.pyramid_type {
      ImagePyramidType::Lowpass => {
        let mut pyramid = ImagePyramid::with(params);
        pyramid.levels.push(self.0.clone());
        let filter_type: image::imageops::FilterType = match params.smoothing_type {
          SmoothingType::Gaussian => image::imageops::FilterType::Gaussian,
          SmoothingType::Box => return Err("Box filter not yet implemented."),
          SmoothingType::Triangle => image::imageops::FilterType::Triangle
        };
        let mut current_level = self.0.clone();
        while current_level.width() > 1 && current_level.height() > 1 {
          current_level = current_level.resize(
            (current_level.width() as f32 * params.scale_factor) as u32,
            (current_level.height() as f32 * params.scale_factor) as u32,
            filter_type,
          );
          pyramid.levels.push(current_level.clone());
        }
        Ok(pyramid)
      },
      ImagePyramidType::Bandpass => {
        Err("Bandpass pyramid not yet implemented")
      },
      ImagePyramidType::Steerable => {
        Err("Steerable pyramid not yet implemented")
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn image_pyramid_create_rgb8() {
    let image = DynamicImage::new_rgb8(640, 480);

    let pyramid = ImagePyramid::create(&image, &ImagePyramidParams::default()).unwrap();
    assert_eq!(pyramid.levels.len(), 9);
  }

  #[test]
  fn compute_image_pyramid_rgb8() {
    let image = DynamicImage::new_rgb8(640, 480);
    let ipr = ImageToProcess(&image);
    let params = ImagePyramidParams {
      scale_factor: 0.5,
      pyramid_type: ImagePyramidType::Lowpass,
      smoothing_type: SmoothingType::Gaussian
    };

    let pyramid = ipr.compute_image_pyramid(&params).unwrap();
    assert_eq!(pyramid.levels.len(), 9);
  }

  #[test]
  fn image_pyramid_create_rgba8() {
    let image = DynamicImage::new_rgba8(640, 480);

    let pyramid = ImagePyramid::create(&image, &ImagePyramidParams::default()).unwrap();
    assert_eq!(pyramid.levels.len(), 9);
  }

  #[test]
  fn compute_image_pyramid_rgba8() {
    let image = DynamicImage::new_rgba8(640, 480);
    let ipr = ImageToProcess(&image);
    let params = ImagePyramidParams {
      scale_factor: 0.5,
      pyramid_type: ImagePyramidType::Lowpass,
      smoothing_type: SmoothingType::Gaussian
    };

    let pyramid = ipr.compute_image_pyramid(&params).unwrap();
    assert_eq!(pyramid.levels.len(), 9);
  }

}