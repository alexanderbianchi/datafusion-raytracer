//! # DataFusion Ray Tracer
//!
//! A ray tracer implemented using DataFusion's DataFrame API and SQL queries.
//! Every pixel is computed as a row in a DataFrame -- no traditional rendering
//! loops, just query planning and columnar execution.
//!
//! ## Usage
//! ```bash
//! cargo run --release -- [scene_name]
//! ```
//!
//! Output images are written as PPM files to the current directory.

mod antialiased;
mod dataframe_sphere;
mod gradient;
mod lit_scene;
mod multi_sphere;
mod ppm;
mod reflections;
mod refraction;
mod shadows;
mod sphere;

use datafusion::error::{DataFusionError, Result};
use strum::{IntoEnumIterator, VariantNames};
use strum_macros::{Display, EnumIter, EnumString, VariantNames};

#[derive(EnumIter, EnumString, Display, VariantNames)]
#[strum(serialize_all = "snake_case")]
enum Scene {
    All,
    Gradient,
    Sphere,
    DataframeSphere,
    MultiSphere,
    LitScene,
    Shadows,
    Reflections,
    Refraction,
    Antialiased,
}

impl Scene {
    fn runnable() -> impl Iterator<Item = Scene> {
        Scene::iter().filter(|v| !matches!(v, Scene::All))
    }

    async fn run(&self) -> Result<()> {
        match self {
            Scene::All => {
                for scene in Scene::runnable() {
                    println!("Rendering scene: {scene}");
                    Box::pin(scene.run()).await?;
                }
            }
            Scene::Gradient => gradient::render().await?,
            Scene::Sphere => sphere::render().await?,
            Scene::DataframeSphere => dataframe_sphere::render().await?,
            Scene::MultiSphere => multi_sphere::render().await?,
            Scene::LitScene => lit_scene::render().await?,
            Scene::Shadows => shadows::render().await?,
            Scene::Reflections => reflections::render().await?,
            Scene::Refraction => refraction::render().await?,
            Scene::Antialiased => antialiased::render().await?,
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let usage = format!(
        "Usage: cargo run --release -- [{}]",
        Scene::VARIANTS.join("|")
    );

    let scene: Scene = std::env::args()
        .nth(1)
        .ok_or_else(|| DataFusionError::Execution(format!("Missing argument. {usage}")))?
        .parse()
        .map_err(|_| DataFusionError::Execution(format!("Unknown scene. {usage}")))?;

    scene.run().await
}
