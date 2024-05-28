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

//! `cargo run --example compute_pyramid`

use clap::Parser;
use image_pyramid::*;

#[derive(Parser, Debug)]
#[command(
  version,
  about,
  long_about = "Compute the image pyramid for the given image file and saves the resultant image \
                as files in the specified directory"
)]
struct Args {
  /// Path to an image for which to compute a pyramid
  #[arg(long, value_name = "STR")]
  input: String,

  /// Path to a directory where result files will be saved
  #[arg(long, value_name = "STR")]
  output: String,

  /// Type of pyramid to compute.
  #[arg(long, value_name = "STR", default_value = "lowpass")]
  pyramid_type: Option<String>,

  /// Type of smoothing to use when computing the pyramid.
  #[arg(long, value_name = "STR", default_value = "gaussian")]
  smoothing_type: Option<String>,

  /// The scale factor when computing the pyramid. Must be in the range (0, 1)
  #[arg(long, value_name = "FLOAT", default_value = "0.5")]
  scale_factor: Option<f32>,

  /// The size of the kernel to use when computing the pyramid. Must be an odd number.
  #[arg(long, value_name = "UINT", default_value = "3")]
  kernel_size: Option<u8>,

  /// The number of orientations to use when computing a steerable pyramid. Only used
  /// when `pyramid_type` is set to "steerable". If unspecified, defaults to 4.
  #[arg(long, value_name = "UINT", default_value = "4")]
  num_orientations: Option<u8>,
}

fn main() {
  let args = Args::parse();
  dbg!(&args);
  let image = match image::open(&args.input) {
    Ok(image) => image,
    Err(e) => {
      eprintln!("Error opening image: {}", e);
      return;
    }
  };
  let image_extension = match std::path::Path::new(&args.input).extension() {
    Some(ext) => ext.to_str().unwrap(),
    None => {
      eprintln!("Error getting image extension");
      return;
    }
  };
  let mut params = ImagePyramidParams::default();
  let kernel_size = if let Some(kernel_size) = args.kernel_size {
    kernel_size.into_odd_value().unwrap_or(OddValue::new(3).unwrap())
  } else {
    OddValue::new(3).unwrap()
  };
  let smoothing_type = if let Some(smoothing_type) = args.smoothing_type {
    match smoothing_type.to_lowercase().as_str() {
      "gaussian" => SmoothingType::Gaussian,
      "box" => SmoothingType::Box,
      "triangle" => SmoothingType::Triangle,
      _ => {
        eprintln!(
          "Invalid smoothing type: {}. Defaulting to gaussian",
          smoothing_type
        );
        SmoothingType::Gaussian
      }
    }
  } else {
    SmoothingType::Gaussian
  };
  params.pyramid_type = if let Some(pyramid_type) = args.pyramid_type {
    match pyramid_type.to_lowercase().as_str() {
      "laplacian" | "bandpass" => ImagePyramidType::Bandpass(smoothing_type(kernel_size)),
      "gaussian" | "lowpass" => ImagePyramidType::Lowpass(smoothing_type(kernel_size)),
      "steerable" => {
        let orientations = if let Some(orientations) = args.num_orientations {
          eprintln!("Number of orientations must be specified when using steerable pyramid");
          NonZeroU8::new(orientations)
        } else {
          NonZeroU8::new(4)
        };
        let orientations = if let Some(orientations) = orientations { orientations } else {
          eprintln!("Invalid number of orientations: {}. Defaulting to 4", args.num_orientations.unwrap());
          return;
        };
        ImagePyramidType::Steerable(SteerableParams::new(orientations, kernel_size))
      }
      _ => {
        eprintln!(
          "Invalid pyramid type: {}. Defaulting to gaussian (lowpass)",
          pyramid_type
        );
        ImagePyramidType::Lowpass(smoothing_type(kernel_size))
      }
    }
  } else {
    ImagePyramidType::Lowpass(smoothing_type(kernel_size))
  };

  params.scale_factor = if let Some(scale_factor) = args.scale_factor {
    match scale_factor.into_unit_interval() {
      Ok(f) => f,
      Err(_) => {
        eprintln!("Invalid scale factor: {}. Defaulting to 0.5", scale_factor);
        0.5.into_unit_interval().unwrap()
      }
    }
  } else {
    0.5.into_unit_interval().unwrap()
  };
  let pyramid = match ImagePyramid::create(&image, Some(&params)) {
    Ok(pyramid) => pyramid,
    Err(e) => {
      eprintln!("Error creating image pyramid: {}", e);
      return;
    }
  };

  let save_image_and_log = |image: &image::DynamicImage, filename: &str| {
    match image.save(filename) {
      Ok(_) => println!("Saved image to {}", filename),
      Err(e) => eprintln!("Error saving image: {}", e),
    }
  };

  for (l, level) in pyramid.levels.iter().enumerate() {
    let filename = std::path::Path::new(&args.output).join(format!("L{}.{}", l, image_extension));
    match level {
      ImagePyramidLevel::Single(i) => save_image_and_log(i, filename.to_str().unwrap()),
      ImagePyramidLevel::Bank(images) => {
        for (j, i) in images.iter().enumerate() {
          let filename = std::path::Path::new(&args.output).join(format!("L{}_{}.{}", l, j, image_extension));
          save_image_and_log(i, filename.to_str().unwrap())
        }
      }
    }
  }
}
