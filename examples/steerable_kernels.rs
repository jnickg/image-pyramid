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

//! `cargo run --example steerable kernels -- --output /path/to/output/dir
//! --kernel-size 3 --num-orientations 4`

use anyhow::{Ok, Result};
use clap::Parser;
use image::{DynamicImage, ImageBuffer};
use image_pyramid::*;

#[derive(Parser, Debug)]
#[command(version, about, long_about = "TODO")]
struct Args {
  /// Path to a directory where result files will be saved
  #[arg(long, value_name = "STR")]
  output: String,

  /// The size of the kernel to use when computing the pyramid. Must be an odd
  /// number.
  #[arg(long, value_name = "UINT", default_value = "3")]
  kernel_size: u8,

  /// The number of orientations to use when computing a steerable pyramid.
  #[arg(long, value_name = "UINT", default_value = "4")]
  num_orientations: u8,

  /// Whether to include the basis kernels in the output
  #[arg(long, action)]
  include_basis: bool,
}

fn save_kernel_as_image(kernel: &Kernel<f32>, path: &str) -> Result<()> {
  // first, normalize the kernel data so that its max value is 1.0 and min value is 0.0
  let max_value = kernel.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
  let min_value = kernel.data.iter().cloned().fold(f32::INFINITY, f32::min);
  let kernel_data_normalized: Vec<f32> = kernel.data.iter().map(|&x| (x - min_value) / (max_value - min_value)).collect();
  dbg!((&path, kernel));
  // let kernel_data_normalized = kernel.data.clone();
  let kernel_data_u8 = kernel_data_normalized
    .iter()
    .map(|&x| {
      let x = x as f32;
      let x = x * 255.0;
      x as u8
    })
    .collect();
  let kernel_buffer =
    ImageBuffer::from_raw(kernel.width as u32, kernel.height as u32, kernel_data_u8).unwrap();
  let kernel_image = DynamicImage::ImageLuma8(kernel_buffer);
  kernel_image.save(path)?;
  Ok(())
}

fn main() -> Result<()> {
  let args = Args::parse();
  dbg!(&args);

  let kernel_size = args.kernel_size;
  let num_orientations = args.num_orientations;

  let steerable_params = SteerableParams {
    kernel_size:      OddValue::new(kernel_size)?,
    num_orientations: NonZeroU8::new(num_orientations)
      .ok_or(anyhow::anyhow!("num_orientations must be greater than 0"))?,
  };

  let angles = steerable_params.get_angles();
  dbg!(&angles);

  if args.include_basis {
    let (basis_x, basis_y) = steerable_params.get_basis_kernels()?;
    save_kernel_as_image(&basis_x, &format!("{}/basis_kernel_x.png", args.output))?;
    save_kernel_as_image(&basis_y, &format!("{}/basis_kernel_y.png", args.output))?;
  }

  let kernels = steerable_params.get_kernels()?;
  for (i, kernel) in kernels.iter().enumerate() {
    let kernel_path = format!("{}/kernel_{}.png", args.output, i);
    save_kernel_as_image(kernel, &kernel_path)?;
  }

  Ok(())
}
