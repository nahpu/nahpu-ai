//! Turns an encoded photo into the normalised CHW tensor a CLIP vision tower
//! expects. Mirrors torchvision's Resize(shortest) + CenterCrop on a PIL image.

use std::io::Cursor;

use anyhow::{Context, Result};
use fast_image_resize::images::Image;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, ImageDecoder, ImageReader, RgbImage};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreprocessConfig {
    pub size: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl PreprocessConfig {
    pub const OPENAI_CLIP_224: Self = Self {
        size: 224,
        mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
        std: [0.268_629_54, 0.261_302_58, 0.275_777_11],
    };
}

/// How a non-square photo becomes square.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Framing {
    /// What the models were trained with. Cuts the ends off long specimens.
    CentreCrop,
    /// Keeps the whole photo and fills the gaps with the mean colour, which
    /// normalises to zero.
    PadToSquare,
}

pub fn preprocess(encoded_image: &[u8], config: &PreprocessConfig) -> Result<Vec<f32>> {
    preprocess_framed(encoded_image, config, Framing::CentreCrop)
}

pub fn preprocess_framed(encoded_image: &[u8], config: &PreprocessConfig, framing: Framing) -> Result<Vec<f32>> {
    let upright = decode_upright(encoded_image)?;
    let square = match framing {
        Framing::CentreCrop => resize_and_centre_crop(&upright, config.size)?,
        Framing::PadToSquare => resize_and_pad(&upright, config)?,
    };
    Ok(normalise_to_chw(&square, config))
}

pub fn decode_upright(encoded_image: &[u8]) -> Result<RgbImage> {
    let mut decoder = ImageReader::new(Cursor::new(encoded_image))
        .with_guessed_format()
        .context("could not read the image header")?
        .into_decoder()
        .context("unsupported image format")?;
    let orientation = decoder.orientation().context("could not read orientation")?;

    let mut image = DynamicImage::from_decoder(decoder).context("could not decode the image")?;
    image.apply_orientation(orientation);
    Ok(image.into_rgb8())
}

pub fn resize_and_centre_crop(image: &RgbImage, size: u32) -> Result<RgbImage> {
    let (width, height) = shortest_side_to(size, image.width(), image.height());
    let resized = resize_bicubic(image, width, height)?;

    let left = centre_offset(width, size);
    let top = centre_offset(height, size);
    Ok(image::imageops::crop_imm(&resized, left, top, size, size).to_image())
}

pub fn resize_and_pad(image: &RgbImage, config: &PreprocessConfig) -> Result<RgbImage> {
    let size = config.size;
    let (width, height) = longest_side_to(size, image.width(), image.height());
    let resized = resize_bicubic(image, width, height)?;

    let fill = image::Rgb(config.mean.map(|m| (m * 255.0).round() as u8));
    let mut square = RgbImage::from_pixel(size, size, fill);
    image::imageops::replace(&mut square, &resized, i64::from((size - width) / 2), i64::from((size - height) / 2));
    Ok(square)
}

pub fn normalise_to_chw(square: &RgbImage, config: &PreprocessConfig) -> Vec<f32> {
    let plane = (square.width() * square.height()) as usize;
    let mut chw = vec![0.0; 3 * plane];
    for (index, pixel) in square.pixels().enumerate() {
        for channel in 0..3 {
            let scaled = f32::from(pixel[channel]) / 255.0;
            chw[channel * plane + index] = (scaled - config.mean[channel]) / config.std[channel];
        }
    }
    chw
}

/// Long side is truncated, as torchvision's `int(size * long / short)` does.
fn shortest_side_to(size: u32, width: u32, height: u32) -> (u32, u32) {
    let scaled_long = |long: u32, short: u32| (f64::from(size) * f64::from(long) / f64::from(short)) as u32;
    if width <= height {
        (size, scaled_long(height, width))
    } else {
        (scaled_long(width, height), size)
    }
}

fn longest_side_to(size: u32, width: u32, height: u32) -> (u32, u32) {
    let scaled_short = |short: u32, long: u32| ((f64::from(size) * f64::from(short) / f64::from(long)).round() as u32).max(1);
    if width >= height {
        (size, scaled_short(height, width))
    } else {
        (scaled_short(width, height), size)
    }
}

/// Python's `round` sends halves to the even neighbour, so 36.5 becomes 36.
fn centre_offset(length: u32, size: u32) -> u32 {
    (f64::from(length - size) / 2.0).round_ties_even() as u32
}

/// Catmull-Rom is the a = -0.5 cubic PIL uses for BICUBIC, widened when
/// shrinking the same way PIL does.
fn resize_bicubic(image: &RgbImage, width: u32, height: u32) -> Result<RgbImage> {
    let source = Image::from_vec_u8(image.width(), image.height(), image.as_raw().clone(), PixelType::U8x3)?;
    let mut target = Image::new(width, height, PixelType::U8x3);
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom));
    Resizer::new().resize(&source, &mut target, &options)?;

    RgbImage::from_raw(width, height, target.into_vec()).context("resized buffer has the wrong length")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortest_side_becomes_the_target_and_long_side_truncates() {
        assert_eq!(shortest_side_to(224, 960, 720), (298, 224));
        assert_eq!(shortest_side_to(224, 3200, 2133), (336, 224));
        assert_eq!(shortest_side_to(224, 130, 250), (224, 430));
    }

    #[test]
    fn centre_offset_rounds_halves_to_even_like_python() {
        assert_eq!(centre_offset(297, 224), 36);
        assert_eq!(centre_offset(299, 224), 38);
        assert_eq!(centre_offset(298, 224), 37);
    }

    #[test]
    fn padding_keeps_the_whole_photo_and_centres_it() {
        let config = PreprocessConfig::OPENAI_CLIP_224;
        let wide = RgbImage::from_pixel(400, 200, image::Rgb([255, 255, 255]));

        let square = resize_and_pad(&wide, &config).unwrap();

        assert_eq!(square.dimensions(), (224, 224));
        assert_eq!(square.get_pixel(0, 0), &image::Rgb([123, 117, 104]));
        assert_eq!(square.get_pixel(112, 112), &image::Rgb([255, 255, 255]));
        assert_eq!(square.get_pixel(112, 56), &image::Rgb([255, 255, 255]));
        assert_eq!(square.get_pixel(112, 55), &image::Rgb([123, 117, 104]));
    }

    #[test]
    fn output_is_a_chw_tensor_of_the_configured_size() {
        let photo = RgbImage::from_pixel(400, 300, image::Rgb([200, 100, 50]));
        let config = PreprocessConfig::OPENAI_CLIP_224;

        let square = resize_and_centre_crop(&photo, config.size).unwrap();
        let chw = normalise_to_chw(&square, &config);

        assert_eq!(chw.len(), 3 * 224 * 224);
        let red = (200.0 / 255.0 - config.mean[0]) / config.std[0];
        assert!((chw[0] - red).abs() < 1e-5);
    }
}
