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

  /// Type of pyramid to compute. Default is "gaussian"
  #[arg(long, value_name = "STR", default_value = "gaussian")]
  pyramid_type: Option<String>,

  /// The scale factor when computing the pyramid. Default is 0.5, and must be in the range (0, 1)
  #[arg(long, value_name = "FLOAT", default_value = "0.5")]
  scale_factor: f32,
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
  if let Some(pyramid_type) = args.pyramid_type {
    params.pyramid_type = match pyramid_type.to_lowercase().as_str() {
      "laplacian" | "bandpass" => ImagePyramidType::Bandpass,
      "gaussian" | "lowpass" => ImagePyramidType::Lowpass,
      "steerable" => {
        eprintln!("Steerable pyramid not yet implemented");
        return;
      }
      _ => {
        eprintln!(
          "Invalid pyramid type: {}. Defaulting to gaussian (lowpass)",
          pyramid_type
        );
        ImagePyramidType::Lowpass
      }
    };
  }
  params.scale_factor = match args.scale_factor.into_unit_interval() {
    Ok(scale_factor) => scale_factor,
    Err(_) => {
      eprintln!("Invalid scale factor: {}. Defaulting to 0.5", args.scale_factor);
      0.5.into_unit_interval().unwrap()
    }
  };
  let pyramid = match ImagePyramid::create(&image, Some(&params)) {
    Ok(pyramid) => pyramid,
    Err(e) => {
      eprintln!("Error creating image pyramid: {}", e);
      return;
    }
  };

  for (l, i) in pyramid.levels.iter().enumerate() {
    let filename = std::path::Path::new(&args.output).join(format!("L{}.{}", l, image_extension));
    match i.save(&filename) {
      Ok(_) => println!("Saved image to {}", filename.to_str().unwrap()),
      Err(e) => eprintln!("Error saving image: {}", e),
    }
  }
}
