//! Gradient render - the simplest possible "ray tracer" output.
//!
//! This renders a gradient image using pure SQL (no custom UDFs).
//! It serves as a proof-of-concept that the DataFusion -> PPM pipeline works.
//!
//! The gradient blends from blue (top) to white (bottom), similar to
//! the "sky" background in many ray tracers.

use crate::ppm::write_ppm;
use datafusion::error::Result;
use datafusion::prelude::*;
use std::path::Path;
use std::time::Instant;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

/// Render a gradient image using pure SQL.
pub async fn render() -> Result<()> {
    let ctx = SessionContext::new();
    let start = Instant::now();

    // Generate pixel coordinates and compute gradient colors
    // This demonstrates:
    // - generate_series for creating pixel grid
    // - CROSS JOIN for Cartesian product
    // - Arithmetic expressions for color computation
    let sql = format!(
        r#"
        SELECT
            CAST(x AS INT UNSIGNED) as x,
            CAST(y AS INT UNSIGNED) as y,
            -- Red: gradient from left to right
            CAST((x * 255 / {width}) AS INT UNSIGNED) as r,
            -- Green: slight diagonal gradient
            CAST(((x + y) * 128 / ({width} + {height})) AS INT UNSIGNED) as g,
            -- Blue: gradient from top (255) to bottom (128)
            CAST((255 - (y * 127 / {height})) AS INT UNSIGNED) as b
        FROM
            generate_series(0, {width} - 1) as t1(x),
            generate_series(0, {height} - 1) as t2(y)
        "#,
        width = WIDTH,
        height = HEIGHT
    );

    println!("Executing SQL query for {}x{} image...", WIDTH, HEIGHT);
    let df = ctx.sql(&sql).await?;

    // Collect results
    let batches = df.collect().await?;
    let query_time = start.elapsed();

    let total_pixels: usize = batches.iter().map(|b| b.num_rows()).sum();
    println!(
        "Query returned {} pixels in {:.2?}",
        total_pixels, query_time
    );

    // Write PPM output
    let output_path = Path::new("gradient.ppm");
    write_ppm(&batches, WIDTH, HEIGHT, output_path)?;
    println!("Wrote output to: {}", output_path.display());

    let total_time = start.elapsed();
    println!("Total time: {:.2?}", total_time);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_gradient_render() -> Result<()> {
        // Use smaller dimensions for faster testing
        let ctx = SessionContext::new();

        let sql = r#"
            SELECT
                CAST(x AS INT UNSIGNED) as x,
                CAST(y AS INT UNSIGNED) as y,
                CAST((x * 255 / 10) AS INT UNSIGNED) as r,
                CAST(128 AS INT UNSIGNED) as g,
                CAST((255 - (y * 127 / 10)) AS INT UNSIGNED) as b
            FROM
                generate_series(0, 9) as t1(x),
                generate_series(0, 9) as t2(y)
        "#;

        let df = ctx.sql(sql).await?;
        let batches = df.collect().await?;

        let total_pixels: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_pixels, 100); // 10x10

        // Write to temp file and verify
        let temp_dir = TempDir::new()?;
        let output_path = temp_dir.path().join("test_gradient.ppm");
        write_ppm(&batches, 10, 10, &output_path)?;

        let content = std::fs::read_to_string(&output_path)?;
        assert!(content.starts_with("P3\n10 10\n255\n"));

        Ok(())
    }
}
