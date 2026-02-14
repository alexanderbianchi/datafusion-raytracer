//! DataFrame-based ray tracer - composable and reusable.
//!
//! This demonstrates using DataFusion's DataFrame API instead of raw SQL,
//! making it easy to compose multiple shapes and build complex scenes.

use crate::ppm::write_ppm;
use datafusion::arrow::datatypes::DataType;
use datafusion::common::ScalarValue;
use datafusion::error::Result;
use datafusion::functions::math::expr_fn::sqrt;
use datafusion::logical_expr::{case, col, lit, Expr, ExprSchemable};
use datafusion::prelude::*;
use std::path::Path;
use std::time::Instant;

// Image settings
const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

// Camera settings
const ASPECT_RATIO: f64 = WIDTH as f64 / HEIGHT as f64;
const VIEWPORT_HEIGHT: f64 = 2.0;
const VIEWPORT_WIDTH: f64 = VIEWPORT_HEIGHT * ASPECT_RATIO;
const FOCAL_LENGTH: f64 = 1.0;

/// A sphere in the scene
#[derive(Clone, Debug)]
pub struct Sphere {
    pub center_x: f64,
    pub center_y: f64,
    pub center_z: f64,
    pub radius: f64,
    pub color_r: f64,
    pub color_g: f64,
    pub color_b: f64,
}

impl Sphere {
    pub fn new(cx: f64, cy: f64, cz: f64, radius: f64) -> Self {
        Self {
            center_x: cx,
            center_y: cy,
            center_z: cz,
            radius,
            color_r: 1.0,
            color_g: 0.0,
            color_b: 0.0,
        }
    }

    #[allow(dead_code)]
    pub fn with_color(mut self, r: f64, g: f64, b: f64) -> Self {
        self.color_r = r;
        self.color_g = g;
        self.color_b = b;
        self
    }
}

/// Generate pixel grid as a DataFrame
async fn pixel_grid(ctx: &SessionContext, width: u32, height: u32) -> Result<DataFrame> {
    // Use SQL for generate_series since there's no DataFrame equivalent
    let sql = format!(
        "SELECT CAST(x AS DOUBLE) as x, CAST(y AS DOUBLE) as y \
         FROM generate_series(0, {}) AS t1(x), generate_series(0, {}) AS t2(y)",
        width - 1,
        height - 1
    );
    ctx.sql(&sql).await
}

/// Convert pixel coordinates to rays from camera
fn pixels_to_rays(
    df: DataFrame,
    width: u32,
    height: u32,
    viewport_width: f64,
    viewport_height: f64,
    focal_length: f64,
) -> Result<DataFrame> {
    let w = width as f64;
    let h = height as f64;

    // Camera origin
    let df = df.with_column("origin_x", lit(0.0))?;
    let df = df.with_column("origin_y", lit(0.0))?;
    let df = df.with_column("origin_z", lit(0.0))?;

    // Ray direction through viewport
    let dir_x = (col("x") / lit(w) - lit(0.5)) * lit(viewport_width);
    let dir_y = (lit(0.5) - col("y") / lit(h)) * lit(viewport_height);
    let dir_z = lit(-focal_length);

    let df = df.with_column("dir_x", dir_x)?;
    let df = df.with_column("dir_y", dir_y)?;
    let df = df.with_column("dir_z", dir_z)?;

    Ok(df)
}

/// Compute ray-sphere intersection for a single sphere
/// Returns DataFrame with additional 't' column (NULL if no hit)
fn intersect_sphere(df: DataFrame, sphere: &Sphere, t_column: &str) -> Result<DataFrame> {
    let cx = sphere.center_x;
    let cy = sphere.center_y;
    let cz = sphere.center_z;
    let r = sphere.radius;

    // Quadratic formula coefficients
    // a = dot(dir, dir)
    let a = col("dir_x") * col("dir_x")
        + col("dir_y") * col("dir_y")
        + col("dir_z") * col("dir_z");

    // half_b = dot(dir, origin - center)
    let half_b = col("dir_x") * (col("origin_x") - lit(cx))
        + col("dir_y") * (col("origin_y") - lit(cy))
        + col("dir_z") * (col("origin_z") - lit(cz));

    // c = dot(oc, oc) - radius^2
    let oc_x = col("origin_x") - lit(cx);
    let oc_y = col("origin_y") - lit(cy);
    let oc_z = col("origin_z") - lit(cz);
    let c = oc_x.clone() * oc_x + oc_y.clone() * oc_y + oc_z.clone() * oc_z - lit(r * r);

    // discriminant = half_b^2 - a*c
    let discriminant = half_b.clone() * half_b.clone() - a.clone() * c;

    // t = (-half_b - sqrt(discriminant)) / a, or NULL if discriminant < 0
    let null_f64: Expr = lit(ScalarValue::Float64(None));
    let t = case(discriminant.clone().gt_eq(lit(0.0)))
        .when(
            lit(true),
            (lit(0.0) - half_b.clone() - sqrt(discriminant)) / a,
        )
        .otherwise(null_f64)?;

    df.with_column(t_column, t)
}

/// Compute surface normal at hit point for a sphere
fn compute_normal(
    df: DataFrame,
    sphere: &Sphere,
    t_column: &str,
    prefix: &str,
) -> Result<DataFrame> {
    let cx = sphere.center_x;
    let cy = sphere.center_y;
    let cz = sphere.center_z;
    let r = sphere.radius;
    let t = col(t_column);

    // Hit point P = origin + t * dir
    let hit_x = col("origin_x") + t.clone() * col("dir_x");
    let hit_y = col("origin_y") + t.clone() * col("dir_y");
    let hit_z = col("origin_z") + t * col("dir_z");

    // Normal N = (P - center) / radius
    let df = df.with_column(&format!("{prefix}_normal_x"), (hit_x - lit(cx)) / lit(r))?;
    let df = df.with_column(&format!("{prefix}_normal_y"), (hit_y - lit(cy)) / lit(r))?;
    let df = df.with_column(&format!("{prefix}_normal_z"), (hit_z - lit(cz)) / lit(r))?;

    Ok(df)
}

/// Apply normal-map coloring (visualize surface orientation as RGB)
fn normal_to_color(df: DataFrame, t_column: &str, prefix: &str) -> Result<DataFrame> {
    let t = col(t_column);
    let nx = col(format!("{prefix}_normal_x"));
    let ny = col(format!("{prefix}_normal_y"));
    let nz = col(format!("{prefix}_normal_z"));

    // Check for valid hit (t > 0)
    let hit = t.clone().is_not_null().and(t.gt(lit(0.0)));

    // Map normal [-1, 1] to color [0, 255]
    let r = case(hit.clone())
        .when(lit(true), (nx + lit(1.0)) * lit(0.5) * lit(255.0))
        .otherwise(lit(128.0))?;

    let g = case(hit.clone())
        .when(lit(true), (ny + lit(1.0)) * lit(0.5) * lit(255.0))
        .otherwise(lit(178.0))?;

    let b = case(hit)
        .when(lit(true), (nz + lit(1.0)) * lit(0.5) * lit(255.0))
        .otherwise(lit(255.0))?;

    let df = df.with_column(&format!("{prefix}_r"), r)?;
    let df = df.with_column(&format!("{prefix}_g"), g)?;
    let df = df.with_column(&format!("{prefix}_b"), b)?;

    Ok(df)
}

/// Final projection to output format (x, y, r, g, b as UInt32)
fn to_pixel_output(df: DataFrame, r_col: &str, g_col: &str, b_col: &str) -> Result<DataFrame> {
    let schema = df.schema().clone();
    df.select(vec![
        col("x").cast_to(&DataType::UInt32, &schema)?.alias("x"),
        col("y").cast_to(&DataType::UInt32, &schema)?.alias("y"),
        col(r_col).cast_to(&DataType::UInt32, &schema)?.alias("r"),
        col(g_col).cast_to(&DataType::UInt32, &schema)?.alias("g"),
        col(b_col).cast_to(&DataType::UInt32, &schema)?.alias("b"),
    ])
}

/// Render a scene with a single sphere using DataFrame API
pub async fn render() -> Result<()> {
    let ctx = SessionContext::new();
    let start = Instant::now();

    // Define the scene
    let sphere = Sphere::new(0.0, 0.0, -1.0, 0.5);

    println!(
        "Rendering sphere at ({}, {}, {}) with radius {}",
        sphere.center_x, sphere.center_y, sphere.center_z, sphere.radius
    );
    println!("Using DataFrame API (not raw SQL)");

    // Build the pipeline
    let df = pixel_grid(&ctx, WIDTH, HEIGHT).await?;
    let df = pixels_to_rays(df, WIDTH, HEIGHT, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, FOCAL_LENGTH)?;
    let df = intersect_sphere(df, &sphere, "t")?;
    let df = compute_normal(df, &sphere, "t", "s0")?;
    let df = normal_to_color(df, "t", "s0")?;
    let df = to_pixel_output(df, "s0_r", "s0_g", "s0_b")?;

    // Execute
    let batches = df.collect().await?;
    let query_time = start.elapsed();

    let total_pixels: usize = batches.iter().map(|b| b.num_rows()).sum();
    println!("Rendered {} pixels in {:.2?}", total_pixels, query_time);

    // Write output
    let output_path = Path::new("dataframe_sphere.ppm");
    write_ppm(&batches, WIDTH, HEIGHT, output_path)?;
    println!("Wrote output to: {}", output_path.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pixel_grid() -> Result<()> {
        let ctx = SessionContext::new();
        let df = pixel_grid(&ctx, 10, 10).await?;
        let batches = df.collect().await?;
        let count: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(count, 100);
        Ok(())
    }

    #[tokio::test]
    async fn test_sphere_intersection() -> Result<()> {
        let ctx = SessionContext::new();
        let sphere = Sphere::new(0.0, 0.0, -1.0, 0.5);

        let df = pixel_grid(&ctx, 10, 10).await?;
        let df = pixels_to_rays(df, 10, 10, 2.0, 2.0, 1.0)?;
        let df = intersect_sphere(df, &sphere, "t")?;

        let batches = df.collect().await?;
        assert!(!batches.is_empty());

        // Should have some hits and some misses
        let count: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(count, 100);

        Ok(())
    }

    #[tokio::test]
    async fn test_full_render_pipeline() -> Result<()> {
        let ctx = SessionContext::new();
        let sphere = Sphere::new(0.0, 0.0, -1.0, 0.5);

        let df = pixel_grid(&ctx, 10, 10).await?;
        let df = pixels_to_rays(df, 10, 10, 2.0, 2.0, 1.0)?;
        let df = intersect_sphere(df, &sphere, "t")?;
        let df = compute_normal(df, &sphere, "t", "s0")?;
        let df = normal_to_color(df, "t", "s0")?;
        let df = to_pixel_output(df, "s0_r", "s0_g", "s0_b")?;

        let batches = df.collect().await?;
        let count: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(count, 100);

        // Verify we have the expected columns
        assert_eq!(batches[0].num_columns(), 5); // x, y, r, g, b

        Ok(())
    }
}
