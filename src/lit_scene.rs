//! Properly lit scene with diffuse (Lambertian) shading.
//!
//! This implements real lighting with:
//! - Point light source
//! - Lambertian diffuse shading: intensity = max(0, dot(N, L))
//! - Ambient light for areas not directly lit

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

/// A point light source
#[derive(Clone, Debug)]
pub struct Light {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub intensity: f64,
}

impl Light {
    pub fn new(x: f64, y: f64, z: f64, intensity: f64) -> Self {
        Self { x, y, z, intensity }
    }
}

/// A sphere in the scene
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

/// Generate pixel grid
async fn pixel_grid(ctx: &SessionContext, width: u32, height: u32) -> Result<DataFrame> {
    let sql = format!(
        "SELECT CAST(x AS DOUBLE) as x, CAST(y AS DOUBLE) as y \
         FROM generate_series(0, {}) AS t1(x), generate_series(0, {}) AS t2(y)",
        width - 1,
        height - 1
    );
    ctx.sql(&sql).await
}

/// Convert pixels to rays
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

/// Compute ray-sphere intersection
fn intersect_sphere(df: DataFrame, sphere: &Sphere) -> Result<DataFrame> {
    let cx = sphere.center_x;
    let cy = sphere.center_y;
    let cz = sphere.center_z;
    let r = sphere.radius;
    let name = &sphere.name;

    let a = col("dir_x") * col("dir_x")
        + col("dir_y") * col("dir_y")
        + col("dir_z") * col("dir_z");

    let half_b = col("dir_x") * (col("origin_x") - lit(cx))
        + col("dir_y") * (col("origin_y") - lit(cy))
        + col("dir_z") * (col("origin_z") - lit(cz));

    let oc_x = col("origin_x") - lit(cx);
    let oc_y = col("origin_y") - lit(cy);
    let oc_z = col("origin_z") - lit(cz);
    let c = oc_x.clone() * oc_x + oc_y.clone() * oc_y + oc_z.clone() * oc_z - lit(r * r);

    let discriminant = half_b.clone() * half_b.clone() - a.clone() * c;

    let null_f64: Expr = lit(ScalarValue::Float64(None));
    let t_val = (lit(0.0) - half_b.clone() - sqrt(discriminant.clone())) / a;

    let t = case(discriminant.gt_eq(lit(0.0)))
        .when(lit(true), t_val.clone())
        .otherwise(null_f64.clone())?;

    let t = case(t.clone().gt(lit(0.001)))
        .when(lit(true), t)
        .otherwise(null_f64)?;

    df.with_column(&format!("t_{name}"), t)
}

/// Find nearest hit among all spheres
fn find_nearest_hit(df: DataFrame, spheres: &[Sphere]) -> Result<DataFrame> {
    let mut min_t: Expr = col(&format!("t_{}", spheres[0].name));

    for sphere in spheres.iter().skip(1) {
        let t_col = col(&format!("t_{}", sphere.name));
        min_t = case(min_t.clone().is_null())
            .when(lit(true), t_col.clone())
            .otherwise(
                case(t_col.clone().is_null())
                    .when(lit(true), min_t.clone())
                    .otherwise(
                        case(t_col.clone().lt(min_t.clone()))
                            .when(lit(true), t_col)
                            .otherwise(min_t)?,
                    )?,
            )?;
    }

    df.with_column("t_nearest", min_t)
}

/// Compute which sphere was hit
fn compute_hit_sphere_index(df: DataFrame, spheres: &[Sphere]) -> Result<DataFrame> {
    let t_nearest = col("t_nearest");
    let null_i32: Expr = lit(ScalarValue::Int32(None));

    let mut hit_idx = null_i32.clone();
    for (i, sphere) in spheres.iter().enumerate() {
        let t_col = col(&format!("t_{}", sphere.name));
        let is_hit = t_col
            .clone()
            .is_not_null()
            .and(t_nearest.clone().is_not_null())
            .and(abs(t_col - t_nearest.clone()).lt(lit(0.0001)));

        hit_idx = case(is_hit)
            .when(lit(true), lit(i as i32))
            .otherwise(hit_idx)?;
    }

    df.with_column("hit_sphere", hit_idx)
}

/// Compute hit point position
fn compute_hit_point(df: DataFrame) -> Result<DataFrame> {
    let t = col("t_nearest");

    let hit_x = col("origin_x") + t.clone() * col("dir_x");
    let hit_y = col("origin_y") + t.clone() * col("dir_y");
    let hit_z = col("origin_z") + t * col("dir_z");

    let df = df.with_column("hit_x", hit_x)?;
    let df = df.with_column("hit_y", hit_y)?;
    let df = df.with_column("hit_z", hit_z)?;

    Ok(df)
}

/// Compute normals for all spheres at their respective hit points
fn compute_all_normals(mut df: DataFrame, spheres: &[Sphere]) -> Result<DataFrame> {
    for sphere in spheres {
        let cx = sphere.center_x;
        let cy = sphere.center_y;
        let cz = sphere.center_z;
        let r = sphere.radius;
        let name = &sphere.name;
        let t = col(&format!("t_{name}"));

        let hit_x = col("origin_x") + t.clone() * col("dir_x");
        let hit_y = col("origin_y") + t.clone() * col("dir_y");
        let hit_z = col("origin_z") + t * col("dir_z");

        // Normal = (hit_point - center) / radius
        df = df.with_column(&format!("{name}_nx"), (hit_x - lit(cx)) / lit(r))?;
        df = df.with_column(&format!("{name}_ny"), (hit_y - lit(cy)) / lit(r))?;
        df = df.with_column(&format!("{name}_nz"), (hit_z - lit(cz)) / lit(r))?;
    }
    Ok(df)
}

/// Select the normal from the hit sphere
fn select_hit_normal(df: DataFrame, spheres: &[Sphere]) -> Result<DataFrame> {
    let hit_idx = col("hit_sphere");

    // Default to 0 (will be masked by no-hit check later)
    let mut nx = lit(0.0);
    let mut ny = lit(0.0);
    let mut nz = lit(0.0);

    for (i, sphere) in spheres.iter().enumerate() {
        let name = &sphere.name;
        let is_this = hit_idx.clone().eq(lit(i as i32));

        nx = case(is_this.clone())
            .when(lit(true), col(&format!("{name}_nx")))
            .otherwise(nx)?;
        ny = case(is_this.clone())
            .when(lit(true), col(&format!("{name}_ny")))
            .otherwise(ny)?;
        nz = case(is_this)
            .when(lit(true), col(&format!("{name}_nz")))
            .otherwise(nz)?;
    }

    let df = df.with_column("normal_x", nx)?;
    let df = df.with_column("normal_y", ny)?;
    let df = df.with_column("normal_z", nz)?;

    Ok(df)
}

/// Compute diffuse lighting
/// Lambertian: intensity = max(0, dot(N, L)) where L = normalize(light_pos - hit_pos)
fn compute_diffuse_lighting(df: DataFrame, light: &Light) -> Result<DataFrame> {
    let lx = light.x;
    let ly = light.y;
    let lz = light.z;
    let intensity = light.intensity;

    // Light direction (unnormalized): light_pos - hit_pos
    let to_light_x = lit(lx) - col("hit_x");
    let to_light_y = lit(ly) - col("hit_y");
    let to_light_z = lit(lz) - col("hit_z");

    // Length of light direction vector
    let light_dist_sq = to_light_x.clone() * to_light_x.clone()
        + to_light_y.clone() * to_light_y.clone()
        + to_light_z.clone() * to_light_z.clone();
    let light_dist = sqrt(light_dist_sq);

    // Normalized light direction
    let light_dir_x = to_light_x / light_dist.clone();
    let light_dir_y = to_light_y / light_dist.clone();
    let light_dir_z = to_light_z / light_dist;

    // Dot product: N · L
    let n_dot_l = col("normal_x") * light_dir_x
        + col("normal_y") * light_dir_y
        + col("normal_z") * light_dir_z;

    // Clamp to [0, 1] - only positive values contribute (facing the light)
    let diffuse = case(n_dot_l.clone().gt(lit(0.0)))
        .when(lit(true), n_dot_l)
        .otherwise(lit(0.0))?;

    // Scale by light intensity
    let df = df.with_column("diffuse", diffuse * lit(intensity))?;

    Ok(df)
}

/// Compute final color with ambient + diffuse lighting
fn compute_final_color(df: DataFrame, spheres: &[Sphere], ambient: f64) -> Result<DataFrame> {
    let hit_idx = col("hit_sphere");
    let diffuse = col("diffuse");

    // Sky color for background (gradient from light blue to white)
    let sky_blend = col("y") / lit(HEIGHT as f64);
    let sky_r = lit(255.0) * (lit(1.0) - sky_blend.clone()) + lit(135.0) * sky_blend.clone();
    let sky_g = lit(255.0) * (lit(1.0) - sky_blend.clone()) + lit(206.0) * sky_blend.clone();
    let sky_b = lit(255.0) * (lit(1.0) - sky_blend.clone()) + lit(250.0) * sky_blend;

    let mut final_r = sky_r;
    let mut final_g = sky_g;
    let mut final_b = sky_b;

    for (i, sphere) in spheres.iter().enumerate() {
        let is_this_sphere = hit_idx.clone().eq(lit(i as i32));

        // Color = ambient + diffuse * surface_color
        // Total brightness = ambient + diffuse, clamped to 1.0 max
        let brightness = lit(ambient) + diffuse.clone();
        let brightness = case(brightness.clone().gt(lit(1.0)))
            .when(lit(true), lit(1.0))
            .otherwise(brightness)?;

        let surface_r = lit(sphere.color_r * 255.0) * brightness.clone();
        let surface_g = lit(sphere.color_g * 255.0) * brightness.clone();
        let surface_b = lit(sphere.color_b * 255.0) * brightness;

        final_r = case(is_this_sphere.clone())
            .when(lit(true), surface_r)
            .otherwise(final_r)?;
        final_g = case(is_this_sphere.clone())
            .when(lit(true), surface_g)
            .otherwise(final_g)?;
        final_b = case(is_this_sphere)
            .when(lit(true), surface_b)
            .otherwise(final_b)?;
    }

    // Clamp to 0-255
    let clamp = |e: Expr| -> Result<Expr> {
        Ok(case(e.clone().lt(lit(0.0)))
            .when(lit(true), lit(0.0))
            .otherwise(case(e.clone().gt(lit(255.0))).when(lit(true), lit(255.0)).otherwise(e)?)?)
    };

    let df = df.with_column("final_r", clamp(final_r)?)?;
    let df = df.with_column("final_g", clamp(final_g)?)?;
    let df = df.with_column("final_b", clamp(final_b)?)?;

    Ok(df)
}

/// Final projection to pixel output
fn to_pixel_output(df: DataFrame) -> Result<DataFrame> {
    let schema = df.schema().clone();
    df.select(vec![
        col("x")
            .cast_to(&DataType::UInt32, &schema)?
            .alias("x"),
        col("y")
            .cast_to(&DataType::UInt32, &schema)?
            .alias("y"),
        col("final_r")
            .cast_to(&DataType::UInt32, &schema)?
            .alias("r"),
        col("final_g")
            .cast_to(&DataType::UInt32, &schema)?
            .alias("g"),
        col("final_b")
            .cast_to(&DataType::UInt32, &schema)?
            .alias("b"),
    ])
}

/// Render a properly lit scene
pub async fn render() -> Result<()> {
    let ctx = SessionContext::new();
    let start = Instant::now();

    // Define light source - positioned above and to the left
    let light = Light::new(-2.0, 2.0, 0.0, 1.0);

    // Ambient light level (so shadows aren't pure black)
    let ambient = 0.1;

    // Define spheres
    let spheres = vec![
        // Large red sphere in center
        Sphere::new("red", 0.0, 0.0, -1.5, 0.5).with_color(0.9, 0.2, 0.2),
        // Green sphere to the left
        Sphere::new("green", -1.0, 0.0, -1.2, 0.4).with_color(0.2, 0.9, 0.2),
        // Blue sphere to the right
        Sphere::new("blue", 0.8, 0.0, -1.0, 0.3).with_color(0.2, 0.2, 0.9),
        // Small yellow sphere in front
        Sphere::new("yellow", 0.0, -0.2, -0.7, 0.15).with_color(0.9, 0.9, 0.2),
        // Ground sphere (large sphere below acting as floor)
        Sphere::new("ground", 0.0, -100.5, -1.0, 100.0).with_color(0.5, 0.5, 0.5),
    ];

    println!("Rendering lit scene with {} spheres", spheres.len());
    println!(
        "Light at ({:.1}, {:.1}, {:.1}) intensity={:.1}",
        light.x, light.y, light.z, light.intensity
    );
    println!("Ambient: {:.1}", ambient);
    println!("Spheres:");
    for s in &spheres {
        println!(
            "  {} at ({:.1}, {:.1}, {:.1}) r={:.2} color=({:.1}, {:.1}, {:.1})",
            s.name, s.center_x, s.center_y, s.center_z, s.radius, s.color_r, s.color_g, s.color_b
        );
    }

    // Build the pipeline
    let df = pixel_grid(&ctx, WIDTH, HEIGHT).await?;
    let df = pixels_to_rays(df, WIDTH, HEIGHT, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, FOCAL_LENGTH)?;

    // Intersect all spheres
    let mut df = df;
    for sphere in &spheres {
        df = intersect_sphere(df, sphere)?;
    }

    // Find nearest hit and compute lighting
    let df = find_nearest_hit(df, &spheres)?;
    let df = compute_hit_sphere_index(df, &spheres)?;
    let df = compute_hit_point(df)?;
    let df = compute_all_normals(df, &spheres)?;
    let df = select_hit_normal(df, &spheres)?;
    let df = compute_diffuse_lighting(df, &light)?;
    let df = compute_final_color(df, &spheres, ambient)?;
    let df = to_pixel_output(df)?;

    // Execute
    let batches = df.collect().await?;
    let query_time = start.elapsed();

    let total_pixels: usize = batches.iter().map(|b| b.num_rows()).sum();
    println!("Rendered {} pixels in {:.2?}", total_pixels, query_time);

    // Write output
    let output_path = Path::new("lit_scene.ppm");
    write_ppm(&batches, WIDTH, HEIGHT, output_path)?;
    println!("Wrote output to: {}", output_path.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lit_scene_small() -> Result<()> {
        let ctx = SessionContext::new();
        let light = Light::new(-2.0, 2.0, 0.0, 1.0);
        let ambient = 0.1;

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
        let df = compute_hit_point(df)?;
        let df = compute_all_normals(df, &spheres)?;
        let df = select_hit_normal(df, &spheres)?;
        let df = compute_diffuse_lighting(df, &light)?;
        let df = compute_final_color(df, &spheres, ambient)?;
        let df = to_pixel_output(df)?;

        let batches = df.collect().await?;
        let count: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(count, 100);

        Ok(())
    }
}
