use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use hopfield::net::bitmap::Bitmaps;
use hopfield::net::bitmap_network::train_bitmap;
use hopfield::net::utility::{Mode, SEED, TestEnd, Utils};
use rand::SeedableRng;
use rand::rngs::StdRng;

pub const USAGE: &str = "viz DATASET INDEX hebb|oja sync|async NOISE_PERCENT OUTPUT.html [INPUT]";

pub fn run(args: &[String]) -> Result<(), String> {
    let [dataset, index, rule, update_mode, noise, output, rest @ ..] = args else {
        return Err(USAGE.to_owned());
    };
    if rest.len() > 1 {
        return Err(USAGE.to_owned());
    }
    let index: usize = index
        .parse()
        .map_err(|_| "Index must be a nonnegative integer")?;
    let noise: f32 = noise
        .parse()
        .map_err(|_| "Noise must be a number between 0 and 100")?;
    if !(0.0..=100.0).contains(&noise) {
        return Err("Noise must be a number between 0 and 100".into());
    }
    let mode = match update_mode.as_str() {
        "sync" => Mode::Sync,
        "async" => Mode::Async,
        _ => return Err("Update mode must be sync or async".into()),
    };
    let output = Path::new(output);
    if !output
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("html"))
    {
        return Err("Output filename must end in .html".into());
    }

    let bitmaps = Bitmaps::load(Path::new(dataset))?;
    let (width, height) = (bitmaps.width, bitmaps.height);
    let input_bitmaps = rest
        .first()
        .map(|path| Bitmaps::load(Path::new(path)))
        .transpose()?;
    let input = if let Some(ref bitmaps) = input_bitmaps {
        if (bitmaps.width, bitmaps.height) != (width, height) {
            return Err("Input and training images must have the same dimensions".into());
        }
        &bitmaps.patterns
    } else {
        &bitmaps.patterns
    };
    let pattern = input
        .get(index)
        .ok_or_else(|| format!("Available indices: 0–{}", input.len() - 1))?;

    let (oja, label) = match rule.as_str() {
        "hebb" => (false, "Hebb"),
        "oja" => (true, "Oja"),
        _ => return Err("Learning rule must be hebb or oja".into()),
    };
    let mut probe = Utils::add_noise(pattern, noise / 100.0, &mut StdRng::seed_from_u64(SEED));
    let (history, end) = train_bitmap(&bitmaps.patterns, oja).run(&mut probe, mode);
    let end = match end {
        TestEnd::Stable => "Stable state",
        TestEnd::Cycle => "Two-state cycle",
    };

    let mut html = format!(
        r#"<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hopfield network simulation</title>
<style>
  :root {{ color-scheme: light; font: 16px system-ui, sans-serif; color: #e5edf2; background: #10212b }}
  body {{ margin: 0 auto; max-width: 46rem; padding: 2rem 1rem }}
  h1 {{ margin: 0; font-size: 1.8rem }}
  p {{ color: #bfd0da }}
  figure {{ margin: 1.5rem 0 1rem; padding: 1rem; background: #1a313e; border-radius: .75rem; text-align: center }}
  canvas {{ display: block; margin: auto; width: min(100%, 29rem); height: auto; max-height: 58vh; object-fit: contain; image-rendering: pixelated }}
  .controls {{ display: flex; flex-wrap: wrap; gap: .5rem; margin: 1rem 0 }}
  button {{ padding: .6rem 1rem; border: 0; border-radius: .4rem; font: inherit; color: #10212b; background: #afe6d5; cursor: pointer }}
  button:disabled {{ opacity: .45; cursor: default }}
  input {{ width: 100%; margin: .4rem 0 0; accent-color: #afe6d5 }}
</style>
<main>
  <h1>Hopfield network simulation</h1>
  <p>{width} × {height} neurons · {label} · {update_mode} mode · {noise}% noise</p>
  <figure><canvas id="bitmap" width="{width}" height="{height}" aria-label="Bitmap of the network state"></canvas></figure>
  <label for="frame">Simulation step</label>
  <input id="frame" type="range" min="0" value="0">
  <output id="status" for="frame" aria-live="polite"></output>
  <div class="controls">
    <button id="back">Back</button><button id="next">Next</button><button id="play">Play</button>
  </div>
</main>
<script>
const states = [
"#
    );
    for frame in &history {
        html.push('"');
        html.extend(
            frame
                .iter()
                .map(|&pixel| if pixel > 0.0 { '1' } else { '0' }),
        );
        html.push_str("\",\n");
    }
    writeln!(html, "];\nconst result = '{end}';").expect("writing to a String cannot fail");
    html.push_str(include_str!("viz.js"));
    html.push_str("\n</script>\n</html>\n");
    fs::write(output, html).map_err(|e| format!("{}: {e}", output.display()))?;

    println!(
        "{end} · steps: {} · {}",
        history.len() - 1,
        output.display()
    );
    Ok(())
}
