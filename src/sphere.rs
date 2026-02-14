//! Sphere render - basic ray-sphere intersection using pure SQL.
//!
//! This renders a single sphere using the quadratic formula for ray-sphere
//! intersection, implemented entirely in SQL without custom UDFs.
//!
//! Ray-sphere intersection:
//! Given ray: P(t) = origin + t * direction
//! And sphere: |P - center|^2 = radius^2
//!
//! Solving gives: at^2 + bt + c = 0
//! where:
//!   a = dot(dir, dir)
//!   b = 2 * dot(dir, origin - center)
//!   c = dot(origin - center, origin - center) - radius^2
//!
//! discriminant = b^2 - 4ac
//! If discriminant >= 0, ray hits sphere

use crate::ppm::write_ppm;
use datafusion::error::Result;
use datafusion::prelude::*;
use std::path::Path;
use std::time::Instant;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
const ASPECT_RATIO: f64 = WIDTH as f64 / HEIGHT as f64;

// Camera settings
const VIEWPORT_HEIGHT: f64 = 2.0;
const VIEWPORT_WIDTH: f64 = VIEWPORT_HEIGHT * ASPECT_RATIO;
const FOCAL_LENGTH: f64 = 1.0;

// Sphere settings
const SPHERE_CENTER_X: f64 = 0.0;
const SPHERE_CENTER_Y: f64 = 0.0;
const SPHERE_CENTER_Z: f64 = -1.0;
const SPHERE_RADIUS: f64 = 0.5;

/// Render a sphere using ray-sphere intersection in pure SQL.
pub async fn render() -> Result<()> {
    let ctx = SessionContext::new();
    let start = Instant::now();

    // The SQL query implements the full ray tracing pipeline:
    // 1. Generate pixel coordinates
    // 2. Convert to viewport coordinates (NDC)
    // 3. Create ray directions
    // 4. Compute ray-sphere intersection
    // 5. Calculate surface normal for hit points
    // 6. Color based on normal (normal mapping visualization)
    let sql = format!(
        r#"
        WITH
        -- Generate pixel grid
        pixels AS (
            SELECT
                x,
                y,
                -- Convert pixel coords to viewport coords [-1, 1]
                -- Note: y is flipped (0 at top in image, but +y is up in world)
                (CAST(x AS DOUBLE) / {width} - 0.5) * {viewport_width} AS vp_x,
                (0.5 - CAST(y AS DOUBLE) / {height}) * {viewport_height} AS vp_y
            FROM
                generate_series(0, {width} - 1) AS t1(x),
                generate_series(0, {height} - 1) AS t2(y)
        ),

        -- Create rays from camera origin through viewport
        rays AS (
            SELECT
                x, y,
                -- Camera at origin
                0.0 AS origin_x,
                0.0 AS origin_y,
                0.0 AS origin_z,
                -- Ray direction (not normalized, pointing at viewport pixel)
                vp_x AS dir_x,
                vp_y AS dir_y,
                -{focal_length} AS dir_z
            FROM pixels
        ),

        -- Compute ray-sphere intersection using quadratic formula
        intersections AS (
            SELECT
                x, y,
                origin_x, origin_y, origin_z,
                dir_x, dir_y, dir_z,
                -- Vector from ray origin to sphere center (oc = origin - center)
                origin_x - {cx} AS oc_x,
                origin_y - {cy} AS oc_y,
                origin_z - {cz} AS oc_z,
                -- Quadratic coefficients
                -- a = dot(dir, dir)
                dir_x * dir_x + dir_y * dir_y + dir_z * dir_z AS a,
                -- b = 2 * dot(dir, oc) - we'll compute half_b = dot(dir, oc) for numerical stability
                dir_x * (origin_x - {cx}) + dir_y * (origin_y - {cy}) + dir_z * (origin_z - {cz}) AS half_b,
                -- c = dot(oc, oc) - radius^2
                (origin_x - {cx}) * (origin_x - {cx}) +
                (origin_y - {cy}) * (origin_y - {cy}) +
                (origin_z - {cz}) * (origin_z - {cz}) - {radius} * {radius} AS c
            FROM rays
        ),

        -- Determine hit/miss and compute hit point
        hits AS (
            SELECT
                x, y,
                origin_x, origin_y, origin_z,
                dir_x, dir_y, dir_z,
                a, half_b, c,
                -- discriminant = half_b^2 - a*c (using half_b form)
                half_b * half_b - a * c AS discriminant,
                -- t = (-half_b - sqrt(discriminant)) / a (nearest intersection)
                CASE
                    WHEN half_b * half_b - a * c >= 0
                    THEN (-half_b - sqrt(half_b * half_b - a * c)) / a
                    ELSE NULL
                END AS t
            FROM intersections
        ),

        -- Calculate surface normal at hit point
        normals AS (
            SELECT
                x, y, t, discriminant,
                -- Hit point P = origin + t * dir
                origin_x + t * dir_x AS hit_x,
                origin_y + t * dir_y AS hit_y,
                origin_z + t * dir_z AS hit_z,
                -- Normal N = (P - center) / radius
                (origin_x + t * dir_x - {cx}) / {radius} AS normal_x,
                (origin_y + t * dir_y - {cy}) / {radius} AS normal_y,
                (origin_z + t * dir_z - {cz}) / {radius} AS normal_z
            FROM hits
        ),

        -- Final color computation
        colors AS (
            SELECT
                x, y,
                CASE
                    WHEN t IS NOT NULL AND t > 0 THEN
                        -- Map normal to color: normal is in [-1,1], map to [0,1] then [0,255]
                        -- This creates a nice visualization of the surface orientation
                        CAST((normal_x + 1.0) * 0.5 * 255 AS INT UNSIGNED)
                    ELSE
                        -- Background: sky blue gradient
                        CAST(128 + 127 * (0.5 - CAST(y AS DOUBLE) / {height}) AS INT UNSIGNED)
                END AS r,
                CASE
                    WHEN t IS NOT NULL AND t > 0 THEN
                        CAST((normal_y + 1.0) * 0.5 * 255 AS INT UNSIGNED)
                    ELSE
                        CAST(178 + 77 * (0.5 - CAST(y AS DOUBLE) / {height}) AS INT UNSIGNED)
                END AS g,
                CASE
                    WHEN t IS NOT NULL AND t > 0 THEN
                        CAST((normal_z + 1.0) * 0.5 * 255 AS INT UNSIGNED)
                    ELSE
                        -- Blue stays constant for sky
                        CAST(255 AS INT UNSIGNED)
                END AS b
            FROM normals
        )

        SELECT
            CAST(x AS INT UNSIGNED) AS x,
            CAST(y AS INT UNSIGNED) AS y,
            r, g, b
        FROM colors
        "#,
        width = WIDTH,
        height = HEIGHT,
        viewport_width = VIEWPORT_WIDTH,
        viewport_height = VIEWPORT_HEIGHT,
        focal_length = FOCAL_LENGTH,
        cx = SPHERE_CENTER_X,
        cy = SPHERE_CENTER_Y,
        cz = SPHERE_CENTER_Z,
        radius = SPHERE_RADIUS
    );

    println!("Executing ray-sphere intersection for {}x{} image...", WIDTH, HEIGHT);
    println!("Sphere at ({}, {}, {}) with radius {}",
             SPHERE_CENTER_X, SPHERE_CENTER_Y, SPHERE_CENTER_Z, SPHERE_RADIUS);

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
    let output_path = Path::new("sphere.ppm");
    write_ppm(&batches, WIDTH, HEIGHT, output_path)?;
    println!("Wrote output to: {}", output_path.display());

    let total_time = start.elapsed();
    println!("Total time: {:.2?}", total_time);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::Float64Array;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_sphere_render_small() -> Result<()> {
        // Use small dimensions for fast testing
        let ctx = SessionContext::new();

        // Simplified query for 10x10 image
        let sql = r#"
            WITH
            pixels AS (
                SELECT
                    x, y,
                    (CAST(x AS DOUBLE) / 10 - 0.5) * 2.0 AS vp_x,
                    (0.5 - CAST(y AS DOUBLE) / 10) * 2.0 AS vp_y
                FROM
                    generate_series(0, 9) AS t1(x),
                    generate_series(0, 9) AS t2(y)
            ),
            rays AS (
                SELECT x, y, 0.0 AS ox, 0.0 AS oy, 0.0 AS oz,
                       vp_x AS dx, vp_y AS dy, -1.0 AS dz
                FROM pixels
            ),
            intersections AS (
                SELECT x, y, dx, dy, dz, ox, oy, oz,
                       dx*dx + dy*dy + dz*dz AS a,
                       dx*(ox-0.0) + dy*(oy-0.0) + dz*(oz-(-1.0)) AS half_b,
                       (ox-0.0)*(ox-0.0) + (oy-0.0)*(oy-0.0) + (oz-(-1.0))*(oz-(-1.0)) - 0.5*0.5 AS c
                FROM rays
            ),
            hits AS (
                SELECT x, y,
                       CASE WHEN half_b*half_b - a*c >= 0
                            THEN (-half_b - sqrt(half_b*half_b - a*c)) / a
                            ELSE NULL END AS t
                FROM intersections
            )
            SELECT
                CAST(x AS INT UNSIGNED) AS x,
                CAST(y AS INT UNSIGNED) AS y,
                CAST(CASE WHEN t IS NOT NULL AND t > 0 THEN 255 ELSE 100 END AS INT UNSIGNED) AS r,
                CAST(CASE WHEN t IS NOT NULL AND t > 0 THEN 0 ELSE 149 END AS INT UNSIGNED) AS g,
                CAST(CASE WHEN t IS NOT NULL AND t > 0 THEN 0 ELSE 237 END AS INT UNSIGNED) AS b
            FROM hits
        "#;

        let df = ctx.sql(sql).await?;
        let batches = df.collect().await?;

        let total_pixels: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_pixels, 100);

        // Verify we have some hits (red pixels) and some misses (blue pixels)
        let temp_dir = TempDir::new()?;
        let output_path = temp_dir.path().join("test_sphere.ppm");
        write_ppm(&batches, 10, 10, &output_path)?;

        let content = std::fs::read_to_string(&output_path)?;
        // Should have both red (255 0 0) and blue-ish (100 149 237) pixels
        assert!(content.contains("255 0 0"), "Should have hit pixels (red)");
        assert!(content.contains("100 149 237"), "Should have miss pixels (sky blue)");

        Ok(())
    }

    #[tokio::test]
    async fn test_ray_sphere_intersection_math() -> Result<()> {
        // Test the ray-sphere intersection formula directly
        let ctx = SessionContext::new();

        // Ray from origin pointing at sphere center at z=-1, radius=0.5
        // Should hit at t=0.5 (front of sphere)
        let sql = r#"
            SELECT
                -- Ray: origin (0,0,0), direction (0,0,-1)
                -- Sphere: center (0,0,-1), radius 0.5
                0.0 * 0.0 + 0.0 * 0.0 + (-1.0) * (-1.0) AS a,  -- dot(dir, dir) = 1
                0.0 * 0.0 + 0.0 * 0.0 + (-1.0) * (0.0 - (-1.0)) AS half_b,  -- dot(dir, oc) = -1
                0.0 * 0.0 + 0.0 * 0.0 + (0.0 - (-1.0)) * (0.0 - (-1.0)) - 0.5 * 0.5 AS c,  -- |oc|^2 - r^2 = 1 - 0.25 = 0.75
                (-1.0) * (-1.0) - 1.0 * 0.75 AS discriminant,  -- 1 - 0.75 = 0.25
                (-(-1.0) - sqrt(0.25)) / 1.0 AS t  -- (1 - 0.5) / 1 = 0.5
        "#;

        let df = ctx.sql(sql).await?;
        let batches = df.collect().await?;

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);

        // Extract the t value
        let t_col = batches[0].column(4)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        let t = t_col.value(0);
        // t should be 0.5 (distance to front of sphere)
        assert!((t - 0.5).abs() < 1e-10, "Expected t=0.5, got t={}", t);

        Ok(())
    }
}
