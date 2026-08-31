use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, ExitStatus};

use anyhow::{bail, Context, Result};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "direct-native")]
struct Cli {
    /// Run the hybrid chain with this prefill-range stage left on Raster and shadow-compared.
    #[arg(long = "raster-prefill-range-index")]
    raster_prefill_range_index: Option<usize>,

    /// Compare one already-completed no-auth stage without rerunning the chain.
    #[arg(long = "compare-prefill-range-stage")]
    compare_prefill_range_stage: Option<PathBuf>,

    /// Run one prefill-range stage directly and publish its canonical output artifact.
    #[arg(long = "run-prefill-range-stage", hide = true)]
    run_prefill_range_stage: Option<PathBuf>,

    #[arg(long, hide = true)]
    input: Option<PathBuf>,

    #[arg(long = "input-manifest", hide = true)]
    input_manifest: Option<PathBuf>,
}

fn main() -> ExitCode {
    match try_main() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("direct-native error: {error:#}");
            ExitCode::from(1)
        }
    }
}

fn try_main() -> Result<ExitCode> {
    execute(Cli::parse())
}

fn execute(cli: Cli) -> Result<ExitCode> {
    if let Some(stage_dir) = cli.run_prefill_range_stage.as_ref() {
        let index = cli
            .raster_prefill_range_index
            .ok_or_else(|| anyhow::anyhow!("direct stage mode requires selected index"))?;
        require_input_args(&cli)?;
        direct_native::shadow::run_hidden_stage_direct(stage_dir, index)?;
        return Ok(ExitCode::SUCCESS);
    }

    if let Some(stage_dir) = cli.compare_prefill_range_stage {
        let index = cli
            .raster_prefill_range_index
            .ok_or_else(|| anyhow::anyhow!("comparison mode requires selected index"))?;
        if cli.input.is_none() || cli.input_manifest.is_none() {
            let status = run_hidden_compare(index, &stage_dir)?;
            return finish_shadow_run(status, &stage_dir);
        }
        let report = direct_native::shadow::run_hidden_stage_compare(&stage_dir, index)?;
        return Ok(if report.matched {
            ExitCode::SUCCESS
        } else {
            ExitCode::from(1)
        });
    }

    if let Some(index) = cli.raster_prefill_range_index {
        let exe = std::env::current_exe().context("failed to locate current executable")?;
        let run = direct_native::hybrid::run(index, &exe)?;
        if let Ok(report) = direct_native::shadow::read_shadow_report(&run.selected_stage_dir) {
            print!(
                "{}",
                render_timing_summary_or_warning(&run.chain_dir, &report)
            );
        }
        return Ok(ExitCode::SUCCESS);
    }

    bail!(
        "direct-native requires --raster-prefill-range-index; \
         use the Raster CLI directly for Raster-only execution"
    )
}

fn finish_shadow_run(status: ExitStatus, stage_dir: &Path) -> Result<ExitCode> {
    match direct_native::shadow::read_shadow_report(stage_dir) {
        Ok(report) => {
            let chain_dir = stage_dir
                .parent()
                .ok_or_else(|| anyhow::anyhow!("selected stage has no chain run directory"))?;
            print!("{}", render_timing_summary_or_warning(chain_dir, &report));
        }
        Err(error) if status.success() => return Err(error),
        Err(_) => {}
    }
    Ok(exit_code_from_status(status))
}

fn require_input_args(cli: &Cli) -> Result<()> {
    if cli.input.is_none() || cli.input_manifest.is_none() {
        bail!("hidden stage mode requires --input and --input-manifest");
    }
    Ok(())
}

fn render_timing_summary_or_warning(
    chain_dir: &Path,
    report: &direct_native::shadow::ShadowReport,
) -> String {
    direct_native::shadow::read_chain_execution_times(chain_dir)
        .and_then(|timings| direct_native::shadow::render_timing_summary(&timings, report))
        .unwrap_or_else(|error| format!("\ntiming summary unavailable: {error:#}\n"))
}

fn run_hidden_compare(index: usize, stage_dir: &Path) -> Result<ExitStatus> {
    let input = stage_dir.join("input.json");
    let input_manifest = stage_dir.join("input_manifest.json");
    let prior_report = direct_native::shadow::parity_dir(stage_dir).join("report.json");
    if let Err(error) = std::fs::remove_file(&prior_report) {
        if error.kind() != std::io::ErrorKind::NotFound {
            return Err(error)
                .with_context(|| format!("failed to remove {}", prior_report.display()));
        }
    }
    let exe = std::env::current_exe().context("failed to locate current executable")?;
    let status = comparison_command(&exe, index, stage_dir, &input, &input_manifest)
        .status()
        .context("failed to start direct-native comparison child")?;
    Ok(status)
}

fn comparison_command(
    exe: &Path,
    index: usize,
    stage_dir: &Path,
    input: &Path,
    input_manifest: &Path,
) -> Command {
    let mut command = Command::new(exe);
    command
        .arg("--compare-prefill-range-stage")
        .arg(stage_dir)
        .arg("--raster-prefill-range-index")
        .arg(index.to_string())
        .arg("--input")
        .arg(input)
        .arg("--input-manifest")
        .arg(input_manifest)
        .env(raster_runtime::auth::AUTH_ENV, "0")
        .env_remove(raster_runtime::TRACE_PATH_ENV)
        .env_remove(raster_runtime::TRACE_FORMAT_ENV)
        .env_remove(raster_runtime::OUTPUT_DIR_ENV)
        .env_remove(raster_runtime::PROFILE_PATH_ENV)
        .env_remove(raster_runtime::PROFILE_STREAM_PATH_ENV)
        .env_remove(raster_runtime::PROFILE_RUN_ID_ENV);
    command
}

fn exit_code_from_status(status: ExitStatus) -> ExitCode {
    match status.code() {
        Some(code) => ExitCode::from(code.clamp(0, u8::MAX as i32) as u8),
        None => ExitCode::from(1),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::ffi::{OsStr, OsString};

    use super::*;

    #[test]
    fn bare_invocation_is_rejected() {
        let error = execute(Cli {
            raster_prefill_range_index: None,
            compare_prefill_range_stage: None,
            run_prefill_range_stage: None,
            input: None,
            input_manifest: None,
        })
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("requires --raster-prefill-range-index"));
    }

    #[test]
    fn hidden_direct_stage_requires_input_files() {
        let error = execute(Cli {
            raster_prefill_range_index: Some(3),
            compare_prefill_range_stage: None,
            run_prefill_range_stage: Some(PathBuf::from("prefill_range_l3")),
            input: None,
            input_manifest: None,
        })
        .unwrap_err();

        assert!(error.to_string().contains("--input and --input-manifest"));
    }

    #[test]
    fn comparison_command_is_unauthenticated_and_isolated() {
        let command = comparison_command(
            Path::new("/tmp/direct-native"),
            3,
            Path::new("/tmp/chains-no-auth/run/prefill_range_l3"),
            Path::new("/tmp/input.json"),
            Path::new("/tmp/input_manifest.json"),
        );
        let env: HashMap<OsString, Option<OsString>> = command
            .get_envs()
            .map(|(name, value)| (name.to_owned(), value.map(OsStr::to_owned)))
            .collect();

        assert_eq!(
            env.get(OsStr::new(raster_runtime::auth::AUTH_ENV)),
            Some(&Some(OsString::from("0")))
        );
        for name in [
            raster_runtime::TRACE_PATH_ENV,
            raster_runtime::TRACE_FORMAT_ENV,
            raster_runtime::OUTPUT_DIR_ENV,
            raster_runtime::PROFILE_PATH_ENV,
            raster_runtime::PROFILE_STREAM_PATH_ENV,
            raster_runtime::PROFILE_RUN_ID_ENV,
        ] {
            assert_eq!(env.get(OsStr::new(name)), Some(&None), "{name}");
        }
    }

    #[test]
    fn missing_execution_times_do_not_override_parity() {
        let report = direct_native::shadow::ShadowReport {
            version: 2,
            stage: String::from("prefill_range_l3"),
            index: 3,
            authority: direct_native::shadow::ParityAuthority::NonAuthoritative,
            raster_source_mode: direct_native::shadow::RasterSourceMode::Unauthenticated,
            matched: true,
            input_load_duration_ns: 1,
            kernel_duration_ns: 1,
            encode_write_duration_ns: 1,
            direct_stage_duration_ns: 1,
        };

        let summary = render_timing_summary_or_warning(Path::new("/missing/chain/run"), &report);

        assert!(summary.contains("timing summary unavailable"));
    }
}
