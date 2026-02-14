//! PPM image format writer for ray tracer output.
//!
//! PPM (Portable Pixel Map) is a simple ASCII image format that's easy to generate
//! and can be viewed by most image viewers. Format:
//! ```text
//! P3
//! width height
//! max_color_value
//! r g b r g b ...
//! ```

use datafusion::arrow::array::{Array, UInt32Array};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Write RecordBatches containing pixel data to a PPM file.
///
/// Expects batches with columns: x (u32), y (u32), r (u32), g (u32), b (u32)
/// where r, g, b are in range [0, 255].
///
/// Note: DataFusion may return batches in arbitrary order and partitioned,
/// so we collect all pixels into a buffer first, then write in row order.
pub fn write_ppm(
    batches: &[RecordBatch],
    width: u32,
    height: u32,
    path: &Path,
) -> Result<()> {
    // Pre-allocate pixel buffer (RGB values default to black)
    let mut pixels = vec![(0u8, 0u8, 0u8); (width * height) as usize];

    // Collect pixels from all batches
    for batch in batches {
        let x_col = batch
            .column_by_name("x")
            .expect("missing 'x' column")
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("'x' column must be UInt32");

        let y_col = batch
            .column_by_name("y")
            .expect("missing 'y' column")
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("'y' column must be UInt32");

        let r_col = batch
            .column_by_name("r")
            .expect("missing 'r' column")
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("'r' column must be UInt32");

        let g_col = batch
            .column_by_name("g")
            .expect("missing 'g' column")
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("'g' column must be UInt32");

        let b_col = batch
            .column_by_name("b")
            .expect("missing 'b' column")
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("'b' column must be UInt32");

        for i in 0..batch.num_rows() {
            if x_col.is_null(i) || y_col.is_null(i) {
                continue;
            }

            let x = x_col.value(i);
            let y = y_col.value(i);

            if x >= width || y >= height {
                continue;
            }

            let idx = (y * width + x) as usize;
            pixels[idx] = (
                r_col.value(i).min(255) as u8,
                g_col.value(i).min(255) as u8,
                b_col.value(i).min(255) as u8,
            );
        }
    }

    // Write PPM file
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Header
    writeln!(writer, "P3")?;
    writeln!(writer, "{} {}", width, height)?;
    writeln!(writer, "255")?;

    // Pixel data (row by row, top to bottom)
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let (r, g, b) = pixels[idx];
            write!(writer, "{} {} {} ", r, g, b)?;
        }
        writeln!(writer)?;
    }

    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::UInt32Array;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;
    use tempfile::TempDir;

    fn create_test_batch(pixels: &[(u32, u32, u32, u32, u32)]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::UInt32, false),
            Field::new("y", DataType::UInt32, false),
            Field::new("r", DataType::UInt32, false),
            Field::new("g", DataType::UInt32, false),
            Field::new("b", DataType::UInt32, false),
        ]));

        let x: Vec<u32> = pixels.iter().map(|p| p.0).collect();
        let y: Vec<u32> = pixels.iter().map(|p| p.1).collect();
        let r: Vec<u32> = pixels.iter().map(|p| p.2).collect();
        let g: Vec<u32> = pixels.iter().map(|p| p.3).collect();
        let b: Vec<u32> = pixels.iter().map(|p| p.4).collect();

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt32Array::from(x)),
                Arc::new(UInt32Array::from(y)),
                Arc::new(UInt32Array::from(r)),
                Arc::new(UInt32Array::from(g)),
                Arc::new(UInt32Array::from(b)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_write_ppm_simple() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test.ppm");

        // 2x2 image: red, green, blue, white
        let batch = create_test_batch(&[
            (0, 0, 255, 0, 0),   // top-left: red
            (1, 0, 0, 255, 0),   // top-right: green
            (0, 1, 0, 0, 255),   // bottom-left: blue
            (1, 1, 255, 255, 255), // bottom-right: white
        ]);

        write_ppm(&[batch], 2, 2, &output_path).unwrap();

        let content = std::fs::read_to_string(&output_path).unwrap();
        assert!(content.starts_with("P3\n2 2\n255\n"));
        assert!(content.contains("255 0 0")); // red
        assert!(content.contains("0 255 0")); // green
        assert!(content.contains("0 0 255")); // blue
        assert!(content.contains("255 255 255")); // white
    }

    #[test]
    fn test_write_ppm_unordered_batches() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test.ppm");

        // Simulate DataFusion returning pixels out of order across batches
        let batch1 = create_test_batch(&[
            (1, 1, 255, 255, 255), // bottom-right first
            (0, 0, 255, 0, 0),     // top-left
        ]);
        let batch2 = create_test_batch(&[
            (0, 1, 0, 0, 255), // bottom-left
            (1, 0, 0, 255, 0), // top-right
        ]);

        write_ppm(&[batch1, batch2], 2, 2, &output_path).unwrap();

        let content = std::fs::read_to_string(&output_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        // Header
        assert_eq!(lines[0], "P3");
        assert_eq!(lines[1], "2 2");
        assert_eq!(lines[2], "255");

        // First row should be: red, green (regardless of batch order)
        assert!(lines[3].contains("255 0 0"));
        assert!(lines[3].contains("0 255 0"));
    }
}
