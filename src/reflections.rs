//! Scene with reflections!
//!
//! Implements single-bounce reflections:
//! 1. Compute reflection direction: R = I - 2(I·N)N
//! 2. Cast reflected ray from hit point
//! 3. Find what reflected ray hits
//! 4. Blend reflected color with surface color based on reflectivity

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

const SHADOW_BIAS: f64 = 0.001;
const REFLECT_BIAS: f64 = 0.001;

/// Point light source
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

/// Sphere with reflectivity
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
    pub reflectivity: f64, // 0.0 = matte, 1.0 = mirror
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
            reflectivity: 0.0,
        }
    }

    pub fn with_color(mut self, r: f64, g: f64, b: f64) -> Self {
        self.color_r = r;
        self.color_g = g;
        self.color_b = b;
        self
    }

    pub fn with_reflectivity(mut self, r: f64) -> Self {
        self.reflectivity = r.clamp(0.0, 1.0);
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
fn pixels_to_rays(df: DataFrame, width: u32, height: u32) -> Result<DataFrame> {
    let w = width as f64;
    let h = height as f64;

    let df = df.with_column("ray_ox", lit(0.0))?;
    let df = df.with_column("ray_oy", lit(0.0))?;
    let df = df.with_column("ray_oz", lit(0.0))?;

    // Direction (will be normalized later)
    let dir_x = (col("x") / lit(w) - lit(0.5)) * lit(VIEWPORT_WIDTH);
    let dir_y = (lit(0.5) - col("y") / lit(h)) * lit(VIEWPORT_HEIGHT);
    let dir_z = lit(-FOCAL_LENGTH);

    // Normalize direction
    let len = sqrt(dir_x.clone() * dir_x.clone() + dir_y.clone() * dir_y.clone() + dir_z.clone() * dir_z.clone());

    let df = df.with_column("ray_dx", dir_x / len.clone())?;
    let df = df.with_column("ray_dy", dir_y / len.clone())?;
    let df = df.with_column("ray_dz", dir_z / len)?;

    Ok(df)
}

/// Generic ray-sphere intersection
/// Takes column prefixes for ray origin (ox, oy, oz) and direction (dx, dy, dz)
fn intersect_sphere_generic(
    df: DataFrame,
    sphere: &Sphere,
    ray_prefix: &str,
    result_prefix: &str,
) -> Result<DataFrame> {
    let cx = sphere.center_x;
    let cy = sphere.center_y;
    let cz = sphere.center_z;
    let r = sphere.radius;

    let ox = col(&format!("{ray_prefix}_ox"));
    let oy = col(&format!("{ray_prefix}_oy"));
    let oz = col(&format!("{ray_prefix}_oz"));
    let dx = col(&format!("{ray_prefix}_dx"));
    let dy = col(&format!("{ray_prefix}_dy"));
    let dz = col(&format!("{ray_prefix}_dz"));

    let a = dx.clone() * dx.clone() + dy.clone() * dy.clone() + dz.clone() * dz.clone();

    let half_b = dx.clone() * (ox.clone() - lit(cx))
        + dy.clone() * (oy.clone() - lit(cy))
        + dz.clone() * (oz.clone() - lit(cz));

    let oc_x = ox - lit(cx);
    let oc_y = oy - lit(cy);
    let oc_z = oz - lit(cz);
    let c = oc_x.clone() * oc_x + oc_y.clone() * oc_y + oc_z.clone() * oc_z - lit(r * r);

    let discriminant = half_b.clone() * half_b.clone() - a.clone() * c;

    let null_f64: Expr = lit(ScalarValue::Float64(None));
    let t_val = (lit(0.0) - half_b.clone() - sqrt(discriminant.clone())) / a;

    let t = case(discriminant.gt_eq(lit(0.0)))
        .when(lit(true), t_val)
        .otherwise(null_f64.clone())?;

    let t = case(t.clone().gt(lit(0.001)))
        .when(lit(true), t)
        .otherwise(null_f64)?;

    df.with_column(&format!("{result_prefix}_t_{}", sphere.name), t)
}

/// Find nearest hit among spheres for a given ray type
fn find_nearest_hit_generic(
    df: DataFrame,
    spheres: &[Sphere],
    result_prefix: &str,
) -> Result<DataFrame> {
    let mut min_t: Expr = col(&format!("{result_prefix}_t_{}", spheres[0].name));

    for sphere in spheres.iter().skip(1) {
        let t_col = col(&format!("{result_prefix}_t_{}", sphere.name));
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

    df.with_column(&format!("{result_prefix}_t_nearest"), min_t)
}

/// Compute which sphere was hit
fn compute_hit_index_generic(
    df: DataFrame,
    spheres: &[Sphere],
    result_prefix: &str,
) -> Result<DataFrame> {
    let t_nearest = col(&format!("{result_prefix}_t_nearest"));
    let null_i32: Expr = lit(ScalarValue::Int32(None));

    let mut hit_idx = null_i32.clone();
    for (i, sphere) in spheres.iter().enumerate() {
        let t_col = col(&format!("{result_prefix}_t_{}", sphere.name));
        let is_hit = t_col
            .clone()
            .is_not_null()
            .and(t_nearest.clone().is_not_null())
            .and(abs(t_col - t_nearest.clone()).lt(lit(0.0001)));

        hit_idx = case(is_hit)
            .when(lit(true), lit(i as i32))
            .otherwise(hit_idx)?;
    }

    df.with_column(&format!("{result_prefix}_hit_idx"), hit_idx)
}

/// Compute hit point and normal for primary ray
fn compute_primary_hit(df: DataFrame, spheres: &[Sphere]) -> Result<DataFrame> {
    let t = col("primary_t_nearest");
    let hit_idx = col("primary_hit_idx");

    // Hit point
    let hit_x = col("ray_ox") + t.clone() * col("ray_dx");
    let hit_y = col("ray_oy") + t.clone() * col("ray_dy");
    let hit_z = col("ray_oz") + t * col("ray_dz");

    let df = df.with_column("hit_x", hit_x)?;
    let df = df.with_column("hit_y", hit_y)?;
    let df = df.with_column("hit_z", hit_z)?;

    // Normal based on which sphere was hit
    let mut nx = lit(0.0);
    let mut ny = lit(0.0);
    let mut nz = lit(0.0);

    for (i, sphere) in spheres.iter().enumerate() {
        let is_this = hit_idx.clone().eq(lit(i as i32));
        let cx = sphere.center_x;
        let cy = sphere.center_y;
        let cz = sphere.center_z;
        let r = sphere.radius;

        nx = case(is_this.clone())
            .when(lit(true), (col("hit_x") - lit(cx)) / lit(r))
            .otherwise(nx)?;
        ny = case(is_this.clone())
            .when(lit(true), (col("hit_y") - lit(cy)) / lit(r))
            .otherwise(ny)?;
        nz = case(is_this)
            .when(lit(true), (col("hit_z") - lit(cz)) / lit(r))
            .otherwise(nz)?;
    }

    let df = df.with_column("normal_x", nx)?;
    let df = df.with_column("normal_y", ny)?;
    let df = df.with_column("normal_z", nz)?;

    Ok(df)
}

/// Compute reflection ray direction: R = I - 2(I·N)N
fn compute_reflection_ray(df: DataFrame) -> Result<DataFrame> {
    // Incident direction (normalized primary ray direction)
    let ix = col("ray_dx");
    let iy = col("ray_dy");
    let iz = col("ray_dz");

    // Normal
    let nx = col("normal_x");
    let ny = col("normal_y");
    let nz = col("normal_z");

    // I·N (dot product)
    let i_dot_n = ix.clone() * nx.clone() + iy.clone() * ny.clone() + iz.clone() * nz.clone();

    // R = I - 2(I·N)N
    let ref_dx = ix - lit(2.0) * i_dot_n.clone() * nx.clone();
    let ref_dy = iy - lit(2.0) * i_dot_n.clone() * ny.clone();
    let ref_dz = iz - lit(2.0) * i_dot_n * nz.clone();

    // Reflection ray origin = hit point + normal * bias
    let ref_ox = col("hit_x") + nx * lit(REFLECT_BIAS);
    let ref_oy = col("hit_y") + ny * lit(REFLECT_BIAS);
    let ref_oz = col("hit_z") + nz * lit(REFLECT_BIAS);

    let df = df.with_column("reflect_ox", ref_ox)?;
    let df = df.with_column("reflect_oy", ref_oy)?;
    let df = df.with_column("reflect_oz", ref_oz)?;
    let df = df.with_column("reflect_dx", ref_dx)?;
    let df = df.with_column("reflect_dy", ref_dy)?;
    let df = df.with_column("reflect_dz", ref_dz)?;

    Ok(df)
}

/// Compute reflected hit point and normal
fn compute_reflect_hit(df: DataFrame, spheres: &[Sphere]) -> Result<DataFrame> {
    let t = col("reflect_t_nearest");
    let hit_idx = col("reflect_hit_idx");

    // Reflected hit point
    let hit_x = col("reflect_ox") + t.clone() * col("reflect_dx");
    let hit_y = col("reflect_oy") + t.clone() * col("reflect_dy");
    let hit_z = col("reflect_oz") + t * col("reflect_dz");

    let df = df.with_column("reflect_hit_x", hit_x)?;
    let df = df.with_column("reflect_hit_y", hit_y)?;
    let df = df.with_column("reflect_hit_z", hit_z)?;

    // Normal at reflected hit
    let mut nx = lit(0.0);
    let mut ny = lit(0.0);
    let mut nz = lit(0.0);

    for (i, sphere) in spheres.iter().enumerate() {
        let is_this = hit_idx.clone().eq(lit(i as i32));
        let cx = sphere.center_x;
        let cy = sphere.center_y;
        let cz = sphere.center_z;
        let r = sphere.radius;

        nx = case(is_this.clone())
            .when(lit(true), (col("reflect_hit_x") - lit(cx)) / lit(r))
            .otherwise(nx)?;
        ny = case(is_this.clone())
            .when(lit(true), (col("reflect_hit_y") - lit(cy)) / lit(r))
            .otherwise(ny)?;
        nz = case(is_this)
            .when(lit(true), (col("reflect_hit_z") - lit(cz)) / lit(r))
            .otherwise(nz)?;
    }

    let df = df.with_column("reflect_normal_x", nx)?;
    let df = df.with_column("reflect_normal_y", ny)?;
    let df = df.with_column("reflect_normal_z", nz)?;

    Ok(df)
}

/// Compute shadow ray and check for shadows
fn compute_shadows(mut df: DataFrame, spheres: &[Sphere], light: &Light, prefix: &str, hit_prefix: &str) -> Result<DataFrame> {
    let hit_x = col(&format!("{hit_prefix}_x"));
    let hit_y = col(&format!("{hit_prefix}_y"));
    let hit_z = col(&format!("{hit_prefix}_z"));
    let normal_x = col(&format!("{prefix}normal_x"));
    let normal_y = col(&format!("{prefix}normal_y"));
    let normal_z = col(&format!("{prefix}normal_z"));

    // Shadow ray origin
    let shadow_ox = hit_x.clone() + normal_x.clone() * lit(SHADOW_BIAS);
    let shadow_oy = hit_y.clone() + normal_y.clone() * lit(SHADOW_BIAS);
    let shadow_oz = hit_z.clone() + normal_z.clone() * lit(SHADOW_BIAS);

    df = df.with_column(&format!("{prefix}shadow_ox"), shadow_ox)?;
    df = df.with_column(&format!("{prefix}shadow_oy"), shadow_oy)?;
    df = df.with_column(&format!("{prefix}shadow_oz"), shadow_oz)?;

    // Direction to light
    let to_light_x = lit(light.x) - hit_x;
    let to_light_y = lit(light.y) - hit_y;
    let to_light_z = lit(light.z) - hit_z;

    let light_dist = sqrt(
        to_light_x.clone() * to_light_x.clone()
            + to_light_y.clone() * to_light_y.clone()
            + to_light_z.clone() * to_light_z.clone(),
    );

    df = df.with_column(&format!("{prefix}shadow_dx"), to_light_x / light_dist.clone())?;
    df = df.with_column(&format!("{prefix}shadow_dy"), to_light_y / light_dist.clone())?;
    df = df.with_column(&format!("{prefix}shadow_dz"), to_light_z / light_dist.clone())?;
    df = df.with_column(&format!("{prefix}light_dist"), light_dist)?;

    // Check shadow intersections
    for sphere in spheres {
        let name = &sphere.name;
        let cx = sphere.center_x;
        let cy = sphere.center_y;
        let cz = sphere.center_z;
        let r = sphere.radius;

        let ox = col(&format!("{prefix}shadow_ox"));
        let oy = col(&format!("{prefix}shadow_oy"));
        let oz = col(&format!("{prefix}shadow_oz"));
        let dx = col(&format!("{prefix}shadow_dx"));
        let dy = col(&format!("{prefix}shadow_dy"));
        let dz = col(&format!("{prefix}shadow_dz"));

        let a = dx.clone() * dx.clone() + dy.clone() * dy.clone() + dz.clone() * dz.clone();
        let half_b = dx * (ox.clone() - lit(cx))
            + dy * (oy.clone() - lit(cy))
            + dz * (oz.clone() - lit(cz));
        let oc_x = ox - lit(cx);
        let oc_y = oy - lit(cy);
        let oc_z = oz - lit(cz);
        let c = oc_x.clone() * oc_x + oc_y.clone() * oc_y + oc_z.clone() * oc_z - lit(r * r);

        let discriminant = half_b.clone() * half_b.clone() - a.clone() * c;
        let null_f64: Expr = lit(ScalarValue::Float64(None));
        let t_val = (lit(0.0) - half_b - sqrt(discriminant.clone())) / a;

        let t = case(discriminant.gt_eq(lit(0.0)))
            .when(lit(true), t_val)
            .otherwise(null_f64.clone())?;
        let t = case(t.clone().gt(lit(0.001)).and(t.clone().lt(col(&format!("{prefix}light_dist")))))
            .when(lit(true), t)
            .otherwise(null_f64)?;

        df = df.with_column(&format!("{prefix}shadow_t_{name}"), t)?;
    }

    // Determine if in shadow
    let mut in_shadow = lit(false);
    for sphere in spheres {
        let t_col = col(&format!("{prefix}shadow_t_{}", sphere.name));
        in_shadow = in_shadow.or(t_col.is_not_null());
    }

    df = df.with_column(&format!("{prefix}in_shadow"), in_shadow)?;

    Ok(df)
}

/// Compute lighting for a hit point
fn compute_lighting_for_hit(
    df: DataFrame,
    light: &Light,
    ambient: f64,
    hit_prefix: &str,
    normal_prefix: &str,
    shadow_col: &str,
    result_col: &str,
) -> Result<DataFrame> {
    let hit_x = col(&format!("{hit_prefix}_x"));
    let hit_y = col(&format!("{hit_prefix}_y"));
    let hit_z = col(&format!("{hit_prefix}_z"));
    let nx = col(&format!("{normal_prefix}_x"));
    let ny = col(&format!("{normal_prefix}_y"));
    let nz = col(&format!("{normal_prefix}_z"));

    // Light direction
    let to_light_x = lit(light.x) - hit_x;
    let to_light_y = lit(light.y) - hit_y;
    let to_light_z = lit(light.z) - hit_z;

    let light_dist = sqrt(
        to_light_x.clone() * to_light_x.clone()
            + to_light_y.clone() * to_light_y.clone()
            + to_light_z.clone() * to_light_z.clone(),
    );

    let lx = to_light_x / light_dist.clone();
    let ly = to_light_y / light_dist.clone();
    let lz = to_light_z / light_dist;

    // N·L
    let n_dot_l = nx * lx + ny * ly + nz * lz;
    let diffuse = case(n_dot_l.clone().gt(lit(0.0)))
        .when(lit(true), n_dot_l * lit(light.intensity))
        .otherwise(lit(0.0))?;

    // If in shadow, no diffuse
    let diffuse = case(col(shadow_col))
        .when(lit(true), lit(0.0))
        .otherwise(diffuse)?;

    let brightness = lit(ambient) + diffuse;
    let brightness = case(brightness.clone().gt(lit(1.0)))
        .when(lit(true), lit(1.0))
        .otherwise(brightness)?;

    df.with_column(result_col, brightness)
}

/// Compute final color with reflections
fn compute_final_color(df: DataFrame, spheres: &[Sphere]) -> Result<DataFrame> {
    let primary_hit_idx = col("primary_hit_idx");
    let reflect_hit_idx = col("reflect_hit_idx");
    let primary_brightness = col("primary_brightness");
    let reflect_brightness = col("reflect_brightness");
    let has_primary_hit = col("primary_t_nearest").is_not_null();
    let _has_reflect_hit = col("reflect_t_nearest").is_not_null();

    // Sky gradient
    let sky_blend = col("y") / lit(HEIGHT as f64);
    let sky_r = lit(255.0) * (lit(1.0) - sky_blend.clone()) + lit(135.0) * sky_blend.clone();
    let sky_g = lit(255.0) * (lit(1.0) - sky_blend.clone()) + lit(206.0) * sky_blend.clone();
    let sky_b = lit(255.0) * (lit(1.0) - sky_blend.clone()) + lit(250.0) * sky_blend;

    // Compute primary surface color
    let mut primary_r = sky_r.clone();
    let mut primary_g = sky_g.clone();
    let mut primary_b = sky_b.clone();
    let mut reflectivity = lit(0.0);

    for (i, sphere) in spheres.iter().enumerate() {
        let is_this = primary_hit_idx.clone().eq(lit(i as i32));

        primary_r = case(is_this.clone())
            .when(lit(true), lit(sphere.color_r * 255.0) * primary_brightness.clone())
            .otherwise(primary_r)?;
        primary_g = case(is_this.clone())
            .when(lit(true), lit(sphere.color_g * 255.0) * primary_brightness.clone())
            .otherwise(primary_g)?;
        primary_b = case(is_this.clone())
            .when(lit(true), lit(sphere.color_b * 255.0) * primary_brightness.clone())
            .otherwise(primary_b)?;
        reflectivity = case(is_this)
            .when(lit(true), lit(sphere.reflectivity))
            .otherwise(reflectivity)?;
    }

    // Compute reflected surface color (or sky if reflection misses)
    let mut reflect_r = sky_r.clone();
    let mut reflect_g = sky_g.clone();
    let mut reflect_b = sky_b.clone();

    for (i, sphere) in spheres.iter().enumerate() {
        let is_this = reflect_hit_idx.clone().eq(lit(i as i32));

        reflect_r = case(is_this.clone())
            .when(lit(true), lit(sphere.color_r * 255.0) * reflect_brightness.clone())
            .otherwise(reflect_r)?;
        reflect_g = case(is_this.clone())
            .when(lit(true), lit(sphere.color_g * 255.0) * reflect_brightness.clone())
            .otherwise(reflect_g)?;
        reflect_b = case(is_this)
            .when(lit(true), lit(sphere.color_b * 255.0) * reflect_brightness.clone())
            .otherwise(reflect_b)?;
    }

    // Blend: final = primary * (1 - reflectivity) + reflected * reflectivity
    // But only if there's a primary hit
    let final_r = case(has_primary_hit.clone())
        .when(
            lit(true),
            primary_r.clone() * (lit(1.0) - reflectivity.clone())
                + reflect_r * reflectivity.clone(),
        )
        .otherwise(sky_r)?;

    let final_g = case(has_primary_hit.clone())
        .when(
            lit(true),
            primary_g.clone() * (lit(1.0) - reflectivity.clone())
                + reflect_g * reflectivity.clone(),
        )
        .otherwise(sky_g)?;

    let final_b = case(has_primary_hit)
        .when(
            lit(true),
            primary_b.clone() * (lit(1.0) - reflectivity.clone()) + reflect_b * reflectivity,
        )
        .otherwise(sky_b)?;

    // Clamp
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

/// Final output
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

/// Render scene with reflections
pub async fn render() -> Result<()> {
    let ctx = SessionContext::new();
    let start = Instant::now();

    let light = Light::new(-3.0, 3.0, 1.0, 1.0);
    let ambient = 0.15;

    // Scene with reflective spheres
    let spheres = vec![
        // Chrome sphere (highly reflective)
        Sphere::new("chrome", -0.6, 0.0, -1.2, 0.4)
            .with_color(0.8, 0.8, 0.8)
            .with_reflectivity(0.8),
        // Red matte sphere
        Sphere::new("red", 0.5, -0.1, -1.5, 0.4)
            .with_color(0.9, 0.1, 0.1)
            .with_reflectivity(0.1),
        // Blue slightly reflective sphere
        Sphere::new("blue", 0.0, -0.3, -0.8, 0.2)
            .with_color(0.1, 0.2, 0.9)
            .with_reflectivity(0.3),
        // Green sphere in back
        Sphere::new("green", -0.3, 0.2, -2.0, 0.35)
            .with_color(0.1, 0.8, 0.2)
            .with_reflectivity(0.2),
        // Reflective ground
        Sphere::new("ground", 0.0, -100.5, -1.0, 100.0)
            .with_color(0.3, 0.3, 0.35)
            .with_reflectivity(0.4),
    ];

    println!("Rendering scene with REFLECTIONS");
    println!("Light at ({:.1}, {:.1}, {:.1})", light.x, light.y, light.z);
    println!("Spheres:");
    for s in &spheres {
        println!(
            "  {} r={:.2} reflect={:.1}",
            s.name, s.radius, s.reflectivity
        );
    }

    // Build pipeline
    let df = pixel_grid(&ctx, WIDTH, HEIGHT).await?;
    let df = pixels_to_rays(df, WIDTH, HEIGHT)?;

    // Primary ray intersections
    let mut df = df;
    for sphere in &spheres {
        df = intersect_sphere_generic(df, sphere, "ray", "primary")?;
    }
    let df = find_nearest_hit_generic(df, &spheres, "primary")?;
    let df = compute_hit_index_generic(df, &spheres, "primary")?;
    let df = compute_primary_hit(df, &spheres)?;

    // Compute reflection ray
    let mut df = compute_reflection_ray(df)?;

    // Reflection ray intersections
    for sphere in &spheres {
        df = intersect_sphere_generic(df, sphere, "reflect", "reflect")?;
    }
    let df = find_nearest_hit_generic(df, &spheres, "reflect")?;
    let df = compute_hit_index_generic(df, &spheres, "reflect")?;
    let df = compute_reflect_hit(df, &spheres)?;

    // Shadows for primary hit
    let df = compute_shadows(df, &spheres, &light, "", "hit")?;

    // Shadows for reflected hit
    let df = compute_shadows(df, &spheres, &light, "reflect_", "reflect_hit")?;

    // Lighting
    let df = compute_lighting_for_hit(df, &light, ambient, "hit", "normal", "in_shadow", "primary_brightness")?;
    let df = compute_lighting_for_hit(df, &light, ambient, "reflect_hit", "reflect_normal", "reflect_in_shadow", "reflect_brightness")?;

    // Final color with reflection blending
    let df = compute_final_color(df, &spheres)?;
    let df = to_pixel_output(df)?;

    let batches = df.collect().await?;
    let query_time = start.elapsed();

    let total_pixels: usize = batches.iter().map(|b| b.num_rows()).sum();
    println!("Rendered {} pixels in {:.2?}", total_pixels, query_time);

    let output_path = Path::new("reflections.ppm");
    write_ppm(&batches, WIDTH, HEIGHT, output_path)?;
    println!("Wrote output to: {}", output_path.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_reflections_small() -> Result<()> {
        let ctx = SessionContext::new();
        let light = Light::new(-2.0, 2.0, 0.0, 1.0);
        let ambient = 0.1;

        let spheres = vec![
            Sphere::new("s1", 0.0, 0.0, -1.0, 0.5)
                .with_color(0.8, 0.8, 0.8)
                .with_reflectivity(0.5),
            Sphere::new("ground", 0.0, -100.5, -1.0, 100.0)
                .with_color(0.5, 0.5, 0.5)
                .with_reflectivity(0.2),
        ];

        let df = pixel_grid(&ctx, 10, 10).await?;
        let df = pixels_to_rays(df, 10, 10)?;

        let mut df = df;
        for sphere in &spheres {
            df = intersect_sphere_generic(df, sphere, "ray", "primary")?;
        }
        let df = find_nearest_hit_generic(df, &spheres, "primary")?;
        let df = compute_hit_index_generic(df, &spheres, "primary")?;
        let df = compute_primary_hit(df, &spheres)?;
        let mut df = compute_reflection_ray(df)?;

        for sphere in &spheres {
            df = intersect_sphere_generic(df, sphere, "reflect", "reflect")?;
        }
        let df = find_nearest_hit_generic(df, &spheres, "reflect")?;
        let df = compute_hit_index_generic(df, &spheres, "reflect")?;
        let df = compute_reflect_hit(df, &spheres)?;

        let df = compute_shadows(df, &spheres, &light, "", "hit")?;
        let df = compute_shadows(df, &spheres, &light, "reflect_", "reflect_hit")?;
        let df = compute_lighting_for_hit(df, &light, ambient, "hit", "normal", "in_shadow", "primary_brightness")?;
        let df = compute_lighting_for_hit(df, &light, ambient, "reflect_hit", "reflect_normal", "reflect_in_shadow", "reflect_brightness")?;
        let df = compute_final_color(df, &spheres)?;
        let df = to_pixel_output(df)?;

        let batches = df.collect().await?;
        let count: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(count, 100);

        Ok(())
    }
}
