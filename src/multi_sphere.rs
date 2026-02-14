//! Multi-sphere scene - demonstrates composability of DataFrame ray tracer.
//!
//! Renders multiple spheres of different sizes, with proper depth ordering
//! (nearest sphere wins).

use crate::ppm::write_ppm;
use datafusion::arrow::datatypes::DataType;
use datafusion::common::ScalarValue;
use datafusion::error::Result;
use datafusion::functions::math::expr_fn::{abs, sqrt};
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

/// A sphere in the scene with color
#[derive(Clone, Debug)]
pub struct Sphere {
    pub name: String,
    pub center_x: f64,
    pub center_y: f64,
    pub center_z: f64,
    pub radius: f64,
    pub color_r: f64,
    pub color_g: f64,
    pub color_b: f64,
}

impl Sphere {
    pub fn new(name: &str, cx: f64, cy: f64, cz: f64, radius: f64) -> Self {
        Self {
            name: name.to_string(),
            center_x: cx,
            center_y: cy,
            center_z: cz,
            radius,
            color_r: 1.0,
            color_g: 1.0,
            color_b: 1.0,
        }
    }

    pub fn with_color(mut self, r: f64, g: f64, b: f64) -> Self {
        self.color_r = r;
        self.color_g = g;
        self.color_b = b;
        self
    }
}

/// Generate pixel grid as a DataFrame
async fn pixel_grid(ctx: &SessionContext, width: u32, height: u32) -> Result<DataFrame> {
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

    let df = df.with_column("origin_x", lit(0.0))?;
    let df = df.with_column("origin_y", lit(0.0))?;
    let df = df.with_column("origin_z", lit(0.0))?;

    let dir_x = (col("x") / lit(w) - lit(0.5)) * lit(viewport_width);
    let dir_y = (lit(0.5) - col("y") / lit(h)) * lit(viewport_height);
    let dir_z = lit(-focal_length);

    let df = df.with_column("dir_x", dir_x)?;
    let df = df.with_column("dir_y", dir_y)?;
    let df = df.with_column("dir_z", dir_z)?;

    Ok(df)
}

/// Compute ray-sphere intersection, returns t value (NULL if no hit)
fn intersect_sphere(df: DataFrame, sphere: &Sphere) -> Result<DataFrame> {
    let cx = sphere.center_x;
    let cy = sphere.center_y;
    let cz = sphere.center_z;
    let r = sphere.radius;
    let name = &sphere.name;

    // a = dot(dir, dir)
    let a = col("dir_x") * col("dir_x")
        + col("dir_y") * col("dir_y")
        + col("dir_z") * col("dir_z");

    // half_b = dot(dir, origin - center)
    let half_b = col("dir_x") * (col("origin_x") - lit(cx))
        + col("dir_y") * (col("origin_y") - lit(cy))
        + col("dir_z") * (col("origin_z") - lit(cz));

    // c = |origin - center|^2 - radius^2
    let oc_x = col("origin_x") - lit(cx);
    let oc_y = col("origin_y") - lit(cy);
    let oc_z = col("origin_z") - lit(cz);
    let c = oc_x.clone() * oc_x + oc_y.clone() * oc_y + oc_z.clone() * oc_z - lit(r * r);

    let discriminant = half_b.clone() * half_b.clone() - a.clone() * c;

    // t = (-half_b - sqrt(discriminant)) / a, or NULL if no hit
    let null_f64: Expr = lit(ScalarValue::Float64(None));
    let t_val = (lit(0.0) - half_b.clone() - sqrt(discriminant.clone())) / a;

    // Only valid if discriminant >= 0 AND t > 0.001 (avoid self-intersection)
    let t = case(discriminant.gt_eq(lit(0.0)))
        .when(lit(true), t_val.clone())
        .otherwise(null_f64.clone())?;

    // Filter out negative t (behind camera)
    let t = case(t.clone().gt(lit(0.001)))
        .when(lit(true), t)
        .otherwise(null_f64)?;

    df.with_column(&format!("t_{name}"), t)
}

/// Find the nearest hit among all spheres
fn find_nearest_hit(df: DataFrame, spheres: &[Sphere]) -> Result<DataFrame> {
    // Build expression for minimum positive t
    // Start with first sphere's t
    let mut min_t: Expr = col(&format!("t_{}", spheres[0].name));

    // Coalesce with each subsequent sphere (take smaller positive value)
    for sphere in spheres.iter().skip(1) {
        let t_col = col(&format!("t_{}", sphere.name));
        // If min_t is null, use t_col; if t_col is null, use min_t; otherwise take minimum
        min_t = case(min_t.clone().is_null())
            .when(lit(true), t_col.clone())
            .otherwise(
                case(t_col.clone().is_null())
                    .when(lit(true), min_t.clone())
                    .otherwise(
                        case(t_col.clone().lt(min_t.clone()))
                            .when(lit(true), t_col)
                            .otherwise(min_t)?
                    )?
            )?;
    }

    df.with_column("t_nearest", min_t)
}

/// Compute which sphere was hit (by index)
fn compute_hit_sphere_index(df: DataFrame, spheres: &[Sphere]) -> Result<DataFrame> {
    let t_nearest = col("t_nearest");
    let null_i32: Expr = lit(ScalarValue::Int32(None));

    // Check each sphere's t against t_nearest
    let mut hit_idx = null_i32.clone();
    for (i, sphere) in spheres.iter().enumerate() {
        let t_col = col(&format!("t_{}", sphere.name));
        // If this sphere's t matches t_nearest (within epsilon), it's the hit
        let is_hit = t_col.clone().is_not_null()
            .and(t_nearest.clone().is_not_null())
            .and(abs(t_col - t_nearest.clone()).lt(lit(0.0001)));

        hit_idx = case(is_hit)
            .when(lit(true), lit(i as i32))
            .otherwise(hit_idx)?;
    }

    df.with_column("hit_sphere", hit_idx)
}

/// Compute normal at hit point for each sphere
fn compute_all_normals(mut df: DataFrame, spheres: &[Sphere]) -> Result<DataFrame> {
    for sphere in spheres {
        let cx = sphere.center_x;
        let cy = sphere.center_y;
        let cz = sphere.center_z;
        let r = sphere.radius;
        let name = &sphere.name;
        let t = col(&format!("t_{name}"));

        // Hit point P = origin + t * dir
        let hit_x = col("origin_x") + t.clone() * col("dir_x");
        let hit_y = col("origin_y") + t.clone() * col("dir_y");
        let hit_z = col("origin_z") + t * col("dir_z");

        // Normal N = (P - center) / radius
        df = df.with_column(&format!("{name}_nx"), (hit_x - lit(cx)) / lit(r))?;
        df = df.with_column(&format!("{name}_ny"), (hit_y - lit(cy)) / lit(r))?;
        df = df.with_column(&format!("{name}_nz"), (hit_z - lit(cz)) / lit(r))?;
    }
    Ok(df)
}

/// Select the final color based on which sphere was hit
fn compute_final_color(df: DataFrame, spheres: &[Sphere]) -> Result<DataFrame> {
    let hit_idx = col("hit_sphere");

    // Background sky color
    let sky_r = lit(135.0);  // Light blue sky
    let sky_g = lit(206.0);
    let sky_b = lit(235.0);

    // Build color selection based on hit sphere
    let mut final_r = sky_r.clone();
    let mut final_g = sky_g.clone();
    let mut final_b = sky_b.clone();

    for (i, sphere) in spheres.iter().enumerate() {
        let name = &sphere.name;
        let is_this_sphere = hit_idx.clone().eq(lit(i as i32));

        // Get normal for this sphere
        let nx = col(&format!("{name}_nx"));
        let ny = col(&format!("{name}_ny"));
        let nz = col(&format!("{name}_nz"));

        // Blend sphere color with normal shading
        // Base color * (0.5 + 0.5 * normal component) gives nice shading
        let shade_r = lit(sphere.color_r * 255.0) * (lit(0.5) + lit(0.5) * nx);
        let shade_g = lit(sphere.color_g * 255.0) * (lit(0.5) + lit(0.5) * ny);
        let shade_b = lit(sphere.color_b * 255.0) * (lit(0.5) + lit(0.5) * nz.clone());

        // Add some blue tint based on z-normal for depth effect
        let shade_b = shade_b + lit(50.0) * (lit(1.0) - nz);

        final_r = case(is_this_sphere.clone())
            .when(lit(true), shade_r)
            .otherwise(final_r)?;
        final_g = case(is_this_sphere.clone())
            .when(lit(true), shade_g)
            .otherwise(final_g)?;
        final_b = case(is_this_sphere)
            .when(lit(true), shade_b)
            .otherwise(final_b)?;
    }

    // Clamp to 0-255
    let clamp = |e: Expr| -> Result<Expr> {
        let clamped = case(e.clone().lt(lit(0.0)))
            .when(lit(true), lit(0.0))
            .otherwise(
                case(e.clone().gt(lit(255.0)))
                    .when(lit(true), lit(255.0))
                    .otherwise(e)?
            )?;
        Ok(clamped)
    };

    let df = df.with_column("final_r", clamp(final_r)?)?;
    let df = df.with_column("final_g", clamp(final_g)?)?;
    let df = df.with_column("final_b", clamp(final_b)?)?;

    Ok(df)
}

/// Final projection to output format
fn to_pixel_output(df: DataFrame) -> Result<DataFrame> {
    let schema = df.schema().clone();
    df.select(vec![
        col("x").cast_to(&DataType::UInt32, &schema)?.alias("x"),
        col("y").cast_to(&DataType::UInt32, &schema)?.alias("y"),
        col("final_r").cast_to(&DataType::UInt32, &schema)?.alias("r"),
        col("final_g").cast_to(&DataType::UInt32, &schema)?.alias("g"),
        col("final_b").cast_to(&DataType::UInt32, &schema)?.alias("b"),
    ])
}

/// Render a scene with multiple spheres
pub async fn render() -> Result<()> {
    let ctx = SessionContext::new();
    let start = Instant::now();

    // Define the scene - multiple spheres of different sizes and colors!
    let spheres = vec![
        // Large red sphere in center
        Sphere::new("red", 0.0, 0.0, -1.5, 0.5)
            .with_color(1.0, 0.2, 0.2),
        // Small green sphere to the left, slightly in front
        Sphere::new("green", -0.7, 0.0, -1.0, 0.25)
            .with_color(0.2, 1.0, 0.2),
        // Medium blue sphere to the right
        Sphere::new("blue", 0.6, -0.2, -1.2, 0.35)
            .with_color(0.2, 0.2, 1.0),
        // Tiny yellow sphere overlapping with red
        Sphere::new("yellow", 0.2, 0.3, -1.0, 0.15)
            .with_color(1.0, 1.0, 0.2),
        // Large ground sphere (like a floor)
        Sphere::new("ground", 0.0, -100.5, -1.0, 100.0)
            .with_color(0.5, 0.5, 0.5),
    ];

    println!("Rendering {} spheres using DataFrame API:", spheres.len());
    for s in &spheres {
        println!("  {} at ({:.1}, {:.1}, {:.1}) r={:.2} color=({:.1}, {:.1}, {:.1})",
                 s.name, s.center_x, s.center_y, s.center_z, s.radius,
                 s.color_r, s.color_g, s.color_b);
    }

    // Build the pipeline
    let df = pixel_grid(&ctx, WIDTH, HEIGHT).await?;
    let df = pixels_to_rays(df, WIDTH, HEIGHT, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, FOCAL_LENGTH)?;

    // Intersect with each sphere
    let mut df = df;
    for sphere in &spheres {
        df = intersect_sphere(df, sphere)?;
    }

    // Find nearest hit and compute colors
    let df = find_nearest_hit(df, &spheres)?;
    let df = compute_hit_sphere_index(df, &spheres)?;
    let df = compute_all_normals(df, &spheres)?;
    let df = compute_final_color(df, &spheres)?;
    let df = to_pixel_output(df)?;

    // Execute
    let batches = df.collect().await?;
    let query_time = start.elapsed();

    let total_pixels: usize = batches.iter().map(|b| b.num_rows()).sum();
    println!("Rendered {} pixels in {:.2?}", total_pixels, query_time);

    // Write output
    let output_path = Path::new("multi_sphere.ppm");
    write_ppm(&batches, WIDTH, HEIGHT, output_path)?;
    println!("Wrote output to: {}", output_path.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_multi_sphere_small() -> Result<()> {
        let ctx = SessionContext::new();

        let spheres = vec![
            Sphere::new("s1", 0.0, 0.0, -1.0, 0.5).with_color(1.0, 0.0, 0.0),
            Sphere::new("s2", 0.5, 0.0, -1.0, 0.3).with_color(0.0, 1.0, 0.0),
        ];

        let df = pixel_grid(&ctx, 10, 10).await?;
        let df = pixels_to_rays(df, 10, 10, 2.0, 2.0, 1.0)?;

        let mut df = df;
        for sphere in &spheres {
            df = intersect_sphere(df, sphere)?;
        }

        let df = find_nearest_hit(df, &spheres)?;
        let df = compute_hit_sphere_index(df, &spheres)?;
        let df = compute_all_normals(df, &spheres)?;
        let df = compute_final_color(df, &spheres)?;
        let df = to_pixel_output(df)?;

        let batches = df.collect().await?;
        let count: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(count, 100);

        Ok(())
    }
}
