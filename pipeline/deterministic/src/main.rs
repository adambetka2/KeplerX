use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::Instant;

const MAGIC: &[u8; 8] = b"FSC_LC2\0";
const HEADER_BYTES: u64 = 32;
const RECORD_BYTES: u64 = 22;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
#[repr(u64)]
enum Photometry {
    PsfFlat = 0,
    PsfQdiag = 1,
    SapQdiag = 2,
    PsfNova = 3,
}

impl Photometry {
    fn from_code(code: u64) -> Result<Self> {
        match code {
            0 => Ok(Self::PsfFlat),
            1 => Ok(Self::PsfQdiag),
            2 => Ok(Self::SapQdiag),
            3 => Ok(Self::PsfNova),
            _ => bail!("unknown FSC_LC2 photometry code {code}"),
        }
    }

    fn from_label(label: &str) -> Result<Self> {
        match label {
            "psf_flat" => Ok(Self::PsfFlat),
            "psf_qdiag" => Ok(Self::PsfQdiag),
            "sap_qdiag" => Ok(Self::SapQdiag),
            "psf_nova" => Ok(Self::PsfNova),
            _ => bail!("unknown photometry label {label}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CurveHeader {
    source_id: u64,
    photometry: Photometry,
    records: usize,
}

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Scan all compact light curves in a directory.
    Scan(ScanArgs),
    /// Audit supplied ephemerides through the event and proposal stages.
    Audit(AuditArgs),
    /// Measure period-specific inverted and scrambled nulls for supplied signals.
    AuditNulls(AuditArgs),
    /// Recover a deterministic injection and reject a deterministic null.
    SelfTest(ConfigArg),
}

#[derive(Args)]
struct ScanArgs {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    input_dir: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    photometry: Option<String>,
}

#[derive(Args)]
struct AuditArgs {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    input_dir: PathBuf,
    #[arg(long)]
    signals: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Args)]
struct ConfigArg {
    #[arg(long)]
    config: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
struct Config {
    protocol_version: String,
    period_min_days: f64,
    period_max_days: f64,
    duration_hours: Vec<f64>,
    event_snr_floor: f64,
    max_events: usize,
    max_pair_harmonic: usize,
    max_period_proposals: usize,
    #[serde(default)]
    priority_period_min_days: Option<f64>,
    #[serde(default)]
    priority_period_max_days: Option<f64>,
    #[serde(default)]
    priority_duration_max_hours: Option<f64>,
    #[serde(default)]
    max_extension_period_proposals: usize,
    #[serde(default)]
    max_extension_signals_per_target: usize,
    #[serde(default)]
    max_residual_signals_per_target: usize,
    period_log_bin_width: f64,
    period_refine_fraction: f64,
    period_refine_steps: usize,
    phase_bins_min: usize,
    phase_bins_max: usize,
    max_signals_per_target: usize,
    #[serde(default)]
    sparse_event_snr_min: Option<f64>,
    #[serde(default)]
    sparse_event_min_points: usize,
    #[serde(default)]
    max_sparse_events_per_target: usize,
    #[serde(default)]
    search_whitening: SearchWhiteningConfig,
    #[serde(default)]
    control_trials: usize,
    #[serde(default)]
    survivor_validation: SurvivorValidationConfig,
    gates: Gates,
}

#[derive(Debug, Clone, Deserialize)]
struct SearchWhiteningConfig {
    #[serde(default)]
    enabled: bool,
    #[serde(default = "default_whitening_window_points")]
    window_points: usize,
    #[serde(default = "default_true")]
    scale_by_quarter: bool,
}

impl Default for SearchWhiteningConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            window_points: default_whitening_window_points(),
            scale_by_quarter: default_true(),
        }
    }
}

fn default_whitening_window_points() -> usize {
    97
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize, Default)]
struct SurvivorValidationConfig {
    #[serde(default)]
    duration_consistency: DurationConsistencyConfig,
    #[serde(default)]
    quarter_loo: QuarterLooConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct DurationConsistencyConfig {
    #[serde(default)]
    enabled: bool,
    #[serde(default = "default_event_snr_min")]
    event_snr_min: f64,
    #[serde(default = "default_min_measurable_events")]
    min_measurable_events: usize,
    #[serde(default = "default_duration_ratio_min")]
    duration_ratio_min: f64,
    #[serde(default = "default_duration_ratio_max")]
    duration_ratio_max: f64,
    #[serde(default = "default_consistent_fraction")]
    min_consistent_fraction: f64,
}

impl Default for DurationConsistencyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            event_snr_min: default_event_snr_min(),
            min_measurable_events: default_min_measurable_events(),
            duration_ratio_min: default_duration_ratio_min(),
            duration_ratio_max: default_duration_ratio_max(),
            min_consistent_fraction: default_consistent_fraction(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct QuarterLooConfig {
    #[serde(default)]
    enabled: bool,
    #[serde(default = "default_loo_period_fraction")]
    period_refine_fraction: f64,
    #[serde(default = "default_loo_period_steps")]
    period_refine_steps: usize,
    #[serde(default = "default_min_testable_quarters")]
    min_testable_quarters: usize,
    #[serde(default = "default_positive_fraction")]
    min_positive_fraction: f64,
    #[serde(default = "default_depth_ratio")]
    min_median_depth_ratio: f64,
    #[serde(default)]
    heldout_snr_min: f64,
}

impl Default for QuarterLooConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            period_refine_fraction: default_loo_period_fraction(),
            period_refine_steps: default_loo_period_steps(),
            min_testable_quarters: default_min_testable_quarters(),
            min_positive_fraction: default_positive_fraction(),
            min_median_depth_ratio: default_depth_ratio(),
            heldout_snr_min: 0.0,
        }
    }
}

fn default_event_snr_min() -> f64 {
    2.0
}

fn default_min_measurable_events() -> usize {
    3
}

fn default_duration_ratio_min() -> f64 {
    0.5
}

fn default_duration_ratio_max() -> f64 {
    2.0
}

fn default_consistent_fraction() -> f64 {
    0.6
}

fn default_loo_period_fraction() -> f64 {
    0.002
}

fn default_loo_period_steps() -> usize {
    5
}

fn default_min_testable_quarters() -> usize {
    2
}

fn default_positive_fraction() -> f64 {
    0.6
}

fn default_depth_ratio() -> f64 {
    0.2
}

#[derive(Debug, Clone, Deserialize)]
struct Gates {
    mes_min: f64,
    n_transits_min: usize,
    n_in_transit_points_min: usize,
    n_quarters_min: usize,
    depth_ppm_min: f64,
    depth_ppm_max: f64,
    odd_even_sigma_max: f64,
    secondary_snr_max: f64,
    single_event_fraction_max: f64,
    red_noise_ratio_max: f64,
    duration_fraction_max: f64,
    #[serde(default)]
    local_mes_min: Option<f64>,
    #[serde(default)]
    local_to_global_depth_min: Option<f64>,
    #[serde(default)]
    v_shape_score_max: Option<f64>,
    #[serde(default)]
    dip_brightening_mes_ratio_min: Option<f64>,
    #[serde(default)]
    dip_control_mes_excess_min: Option<f64>,
}

#[derive(Debug, Clone)]
struct LightCurve {
    source_id: u64,
    photometry: Photometry,
    time: Vec<f64>,
    flux: Vec<f64>,
    error: Vec<f64>,
    quarter: Vec<i16>,
    cadence: Vec<i32>,
}

#[derive(Debug, Clone, Copy)]
struct Event {
    time: f64,
    duration: f64,
    snr: f64,
}

#[derive(Debug, Clone, Copy)]
struct Proposal {
    period: f64,
    event_score: f64,
}

#[derive(Debug, Clone, Copy)]
struct BoxCandidate {
    period: f64,
    t0: f64,
    duration: f64,
    search_mes: f64,
    event_score: f64,
    search_pass: usize,
}

#[derive(Debug, Clone)]
struct Metrics {
    period: f64,
    t0: f64,
    duration: f64,
    depth_ppm: f64,
    mes: f64,
    local_depth_ppm: f64,
    local_mes: f64,
    local_to_global_depth: f64,
    n_transits: usize,
    n_in_transit_points: usize,
    n_quarters: usize,
    odd_even_sigma: f64,
    secondary_snr: f64,
    single_event_fraction: f64,
    red_noise_ratio: f64,
    v_shape_score: f64,
    robust_sigma_ppm: f64,
    period_resolution_days: f64,
}

#[derive(Debug, Clone, Copy)]
struct SingleEventMetrics {
    depth_ppm: f64,
    snr: f64,
    n_in_transit_points: usize,
}

#[derive(Debug, Clone)]
struct DurationConsistency {
    status: &'static str,
    measurable_events: usize,
    consistent_fraction: Option<f64>,
    log2_duration_mad: Option<f64>,
}

impl DurationConsistency {
    fn not_run() -> Self {
        Self {
            status: "NOT_RUN",
            measurable_events: 0,
            consistent_fraction: None,
            log2_duration_mad: None,
        }
    }
}

#[derive(Debug, Clone)]
struct QuarterLoo {
    status: &'static str,
    testable_quarters: usize,
    positive_fraction: Option<f64>,
    median_depth_ratio: Option<f64>,
}

impl QuarterLoo {
    fn not_run() -> Self {
        Self {
            status: "NOT_RUN",
            testable_quarters: 0,
            positive_fraction: None,
            median_depth_ratio: None,
        }
    }
}

#[derive(Debug, Serialize)]
struct SignalRow {
    protocol_version: String,
    source_id: u64,
    photometry: Photometry,
    signal_rank: usize,
    classification: String,
    search_pass: Option<usize>,
    period_days: Option<f64>,
    t0_bkjd: Option<f64>,
    duration_hours: Option<f64>,
    depth_ppm: Option<f64>,
    search_mes: Option<f64>,
    mes: Option<f64>,
    local_depth_ppm: Option<f64>,
    local_mes: Option<f64>,
    local_to_global_depth: Option<f64>,
    n_transits: Option<usize>,
    n_in_transit_points: Option<usize>,
    n_quarters: Option<usize>,
    odd_even_sigma: Option<f64>,
    secondary_snr: Option<f64>,
    single_event_fraction: Option<f64>,
    red_noise_ratio: Option<f64>,
    v_shape_score: Option<f64>,
    robust_sigma_ppm: Option<f64>,
    baseline_days: f64,
    period_resolution_days: Option<f64>,
    proposal_event_score: Option<f64>,
    harmonic_of_rank1: Option<bool>,
    brightening_max_mes: Option<f64>,
    phase_scramble_max_mes: Option<f64>,
    phase_scramble_evaluated: bool,
    dip_brightening_mes_ratio: Option<f64>,
    dip_control_mes_excess: Option<f64>,
    controls_evaluated: bool,
    duration_consistency_status: String,
    duration_measurable_events: usize,
    duration_consistent_fraction: Option<f64>,
    duration_log2_mad: Option<f64>,
    quarter_loo_status: String,
    quarter_loo_testable_quarters: usize,
    quarter_loo_positive_fraction: Option<f64>,
    quarter_loo_median_depth_ratio: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct AuditSignal {
    source_id: u64,
    period_days: f64,
    epoch_bkjd: f64,
    duration_hours: f64,
}

#[derive(Debug, Serialize)]
struct AuditRow {
    source_id: u64,
    period_days: f64,
    epoch_bkjd: f64,
    duration_hours: f64,
    detected_events: usize,
    matched_detected_events: usize,
    matched_event_snr_sum: f64,
    matched_event_snr_max: Option<f64>,
    matched_event_local_snr_max: Option<f64>,
    event_brightening_snr_max: f64,
    proposals: usize,
    exact_proposal_rank: Option<usize>,
    closest_exact_proposal_rank: Option<usize>,
    closest_exact_proposal_error_fraction: Option<f64>,
    harmonic_proposal_rank: Option<usize>,
    harmonic_proposal: Option<i32>,
    closest_harmonic_proposal_rank: Option<usize>,
    closest_harmonic_proposal: Option<i32>,
    closest_harmonic_proposal_error_fraction: Option<f64>,
    center_fold_match_rank: Option<usize>,
    center_fold_match_mes: Option<f64>,
    refined_fold_match_rank: Option<usize>,
    refined_fold_match_mes: Option<f64>,
    fixed_period_search_mes: Option<f64>,
    fixed_period_measured_mes: Option<f64>,
    fixed_period_epoch_error_days: Option<f64>,
}

#[derive(Debug, Serialize)]
struct NullAuditRow {
    source_id: u64,
    period_days: f64,
    epoch_bkjd: f64,
    duration_hours: f64,
    inverted_mes: Option<f64>,
    scramble_mes_0: Option<f64>,
    scramble_mes_1: Option<f64>,
    scramble_mes_2: Option<f64>,
    scramble_mes_3: Option<f64>,
}

struct NullCurve {
    photometry: LightCurve,
    search: LightCurve,
    search_sigma: f64,
    raw_sigma: f64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Scan(args) => run_scan(args),
        Command::Audit(args) => run_audit(args),
        Command::AuditNulls(args) => run_audit_nulls(args),
        Command::SelfTest(args) => run_self_test(&load_config(&args.config)?),
    }
}

fn load_config(path: &Path) -> Result<Config> {
    let data =
        fs::read_to_string(path).with_context(|| format!("reading config {}", path.display()))?;
    let cfg: Config = serde_json::from_str(&data).context("parsing preregistration")?;
    if cfg.period_min_days <= 0.0 || cfg.period_max_days <= cfg.period_min_days {
        bail!("invalid period range")
    }
    match (
        cfg.priority_period_min_days,
        cfg.priority_period_max_days,
        cfg.priority_duration_max_hours,
    ) {
        (Some(minimum), Some(maximum), Some(duration_maximum))
            if minimum >= cfg.period_min_days
                && maximum <= cfg.period_max_days
                && minimum < maximum
                && duration_maximum > 0.0
                && cfg
                    .duration_hours
                    .iter()
                    .any(|duration| *duration <= duration_maximum)
                && cfg.max_extension_period_proposals > 0
                && cfg.max_extension_signals_per_target > 0 => {}
        (None, None, None)
            if cfg.max_extension_period_proposals == 0
                && cfg.max_extension_signals_per_target == 0 => {}
        _ => bail!("invalid priority period range"),
    }
    if cfg.duration_hours.is_empty() || cfg.period_refine_steps < 3 {
        bail!("invalid duration/refinement configuration")
    }
    let sparse_config_valid = match cfg.sparse_event_snr_min {
        Some(threshold) => {
            threshold > 0.0
                && cfg.sparse_event_min_points > 0
                && cfg.max_sparse_events_per_target > 0
        }
        None => cfg.sparse_event_min_points == 0 && cfg.max_sparse_events_per_target == 0,
    };
    if !sparse_config_valid {
        bail!("invalid sparse-event configuration")
    }
    let duration_cfg = &cfg.survivor_validation.duration_consistency;
    if duration_cfg.enabled
        && (duration_cfg.min_measurable_events == 0
            || duration_cfg.duration_ratio_min <= 0.0
            || duration_cfg.duration_ratio_max <= duration_cfg.duration_ratio_min
            || !(0.0..=1.0).contains(&duration_cfg.min_consistent_fraction))
    {
        bail!("invalid transit-duration consistency configuration")
    }
    let loo_cfg = &cfg.survivor_validation.quarter_loo;
    if loo_cfg.enabled
        && (loo_cfg.period_refine_steps < 3
            || loo_cfg.period_refine_steps.is_multiple_of(2)
            || loo_cfg.min_testable_quarters == 0
            || !(0.0..=1.0).contains(&loo_cfg.min_positive_fraction))
    {
        bail!("invalid quarter leave-one-out configuration")
    }
    Ok(cfg)
}

fn run_scan(args: ScanArgs) -> Result<()> {
    let cfg = load_config(&args.config)?;
    let mut inputs = scan_inputs(&args.input_dir)?;
    if let Some(label) = &args.photometry {
        let photometry = Photometry::from_label(label)?;
        inputs.retain(|(_, header)| header.photometry == photometry);
        if inputs.is_empty() {
            bail!(
                "no {label} light curves found in {}",
                args.input_dir.display()
            );
        }
    }
    let sources: HashSet<u64> = inputs.iter().map(|(_, header)| header.source_id).collect();
    eprintln!(
        "protocol={} sources={} views={} threads={}",
        cfg.protocol_version,
        sources.len(),
        inputs.len(),
        rayon::current_num_threads()
    );
    let started = Instant::now();
    let done = AtomicUsize::new(0);
    let nested: Result<Vec<Vec<SignalRow>>> = inputs
        .par_iter()
        .map(|(path, header)| {
            let lc = read_light_curve(path)?;
            if (lc.source_id, lc.photometry, lc.time.len())
                != (header.source_id, header.photometry, header.records)
            {
                bail!("input identity changed after preflight: {}", path.display());
            }
            let rows = scan_light_curve(&lc, &cfg);
            let count = done.fetch_add(1, AtomicOrdering::Relaxed) + 1;
            if count.is_multiple_of(100) || count == inputs.len() {
                eprintln!("completed_views={count}/{}", inputs.len());
            }
            Ok(rows)
        })
        .collect();
    let mut rows: Vec<SignalRow> = nested?.into_iter().flatten().collect();
    reconcile_rows(&inputs, &rows)?;
    rows.sort_by_key(|row| (row.source_id, row.photometry, row.signal_rank));

    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = csv::Writer::from_path(&args.output)
        .with_context(|| format!("creating {}", args.output.display()))?;
    for row in &rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    eprintln!(
        "wrote {} rows to {} in {:.2}s",
        rows.len(),
        args.output.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn audit_harmonic(period: f64, reference: f64) -> Option<i32> {
    if (period / reference - 1.0).abs() <= 0.01 {
        return Some(1);
    }
    (2..=8).find_map(|harmonic| {
        let harmonic = harmonic as f64;
        if (period / (reference * harmonic) - 1.0).abs() <= 0.01 {
            Some(harmonic as i32)
        } else if (period / (reference / harmonic) - 1.0).abs() <= 0.01 {
            Some(-(harmonic as i32))
        } else {
            None
        }
    })
}

fn audit_signal(lc: &LightCurve, cfg: &Config, signal: &AuditSignal) -> AuditRow {
    let search_curve = whiten_for_search(lc, &cfg.search_whitening);
    let search_sigma = robust_sigma(&search_curve.flux);
    let raw_sigma = robust_sigma(&lc.flux);
    let proposal_duration_hours = priority_duration_hours(cfg);
    let events = detect_events(&search_curve, cfg, search_sigma, &proposal_duration_hours);
    let mut matched_events: HashMap<i64, f64> = HashMap::new();
    let mut matched_local_snr_max: Option<f64> = None;
    for event in &events {
        let epoch = ((event.time - signal.epoch_bkjd) / signal.period_days).round() as i64;
        let residual = (event.time - signal.epoch_bkjd - epoch as f64 * signal.period_days).abs();
        let tolerance = 0.75
            * event
                .duration
                .max(signal.duration_hours / 24.0)
                .max(1.0 / 24.0);
        if residual <= tolerance {
            matched_events
                .entry(epoch)
                .and_modify(|snr| *snr = snr.max(event.snr))
                .or_insert(event.snr);
            if let Some(metrics) = measure_single_event(lc, event, raw_sigma) {
                matched_local_snr_max =
                    Some(matched_local_snr_max.map_or(metrics.snr, |snr| snr.max(metrics.snr)));
            }
        }
    }
    let period_max_days = periodic_max_days(lc, cfg);
    let proposals = propose_periods(&events, cfg, period_max_days);
    let exact_proposal_rank = proposals
        .iter()
        .position(|proposal| audit_harmonic(proposal.period, signal.period_days) == Some(1))
        .map(|index| index + 1);
    let closest_exact = proposals
        .iter()
        .enumerate()
        .map(|(index, proposal)| {
            (
                (proposal.period / signal.period_days - 1.0).abs(),
                index + 1,
            )
        })
        .filter(|item| item.0 <= 0.01)
        .min_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));
    let harmonic_match = proposals.iter().enumerate().find_map(|(index, proposal)| {
        audit_harmonic(proposal.period, signal.period_days)
            .filter(|harmonic| *harmonic != 1)
            .map(|harmonic| (index + 1, harmonic))
    });
    let closest_harmonic = proposals
        .iter()
        .enumerate()
        .filter_map(|(index, proposal)| {
            let harmonic = audit_harmonic(proposal.period, signal.period_days)?;
            let target = if harmonic > 0 {
                signal.period_days * harmonic as f64
            } else {
                signal.period_days / -(harmonic as f64)
            };
            Some(((proposal.period / target - 1.0).abs(), index + 1, harmonic))
        })
        .min_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));
    let mut center_folds: Vec<BoxCandidate> = proposals
        .iter()
        .filter_map(|proposal| {
            let durations = period_duration_days(cfg, proposal.period);
            box_search_at_period_with_durations(
                &search_curve,
                cfg,
                search_sigma,
                proposal.period,
                proposal.event_score,
                &durations,
            )
        })
        .collect();
    center_folds.sort_unstable_by(|left, right| {
        right
            .search_mes
            .partial_cmp(&left.search_mes)
            .unwrap_or(Ordering::Equal)
    });
    let cadence = typical_cadence(lc).max(1.0 / 48.94);
    let center_fold_match = center_folds.iter().enumerate().find(|(_, candidate)| {
        audit_harmonic(candidate.period, signal.period_days).is_some()
            && phase_delta(
                candidate.t0,
                signal.epoch_bkjd,
                candidate.period.min(signal.period_days),
            )
            .abs()
                <= 0.5 * candidate.duration.max(signal.duration_hours / 24.0) + cadence
    });
    let baseline = lc.time.last().unwrap() - lc.time.first().unwrap();
    let shortest = cfg.duration_hours[0] / 24.0;
    let middle = (cfg.period_refine_steps / 2) as isize;
    let mut refined_folds = Vec::new();
    for proposal in &proposals {
        let timing_step = (shortest * proposal.period / baseline.max(1.0))
            .max(proposal.period * cfg.period_refine_fraction / cfg.period_refine_steps as f64);
        for step in 0..cfg.period_refine_steps {
            let offset = step as isize - middle;
            let period = proposal.period + offset as f64 * timing_step;
            if period < cfg.period_min_days || period > period_max_days {
                continue;
            }
            let durations = period_duration_days(cfg, period);
            if let Some(candidate) = box_search_at_period_with_durations(
                &search_curve,
                cfg,
                search_sigma,
                period,
                proposal.event_score,
                &durations,
            ) {
                refined_folds.push(candidate);
            }
        }
    }
    refined_folds.sort_unstable_by(|left, right| {
        right
            .search_mes
            .partial_cmp(&left.search_mes)
            .unwrap_or(Ordering::Equal)
    });
    let mut refined_selected: Vec<BoxCandidate> = Vec::new();
    for candidate in refined_folds {
        if refined_selected
            .iter()
            .any(|kept| (candidate.period / kept.period - 1.0).abs() < 0.002)
        {
            continue;
        }
        refined_selected.push(candidate);
    }
    let refined_fold_match = refined_selected.iter().enumerate().find(|(_, candidate)| {
        audit_harmonic(candidate.period, signal.period_days).is_some()
            && phase_delta(
                candidate.t0,
                signal.epoch_bkjd,
                candidate.period.min(signal.period_days),
            )
            .abs()
                <= 0.5 * candidate.duration.max(signal.duration_hours / 24.0) + cadence
    });
    let duration = [signal.duration_hours / 24.0];
    let fixed = box_search_at_period_with_durations(
        &search_curve,
        cfg,
        search_sigma,
        signal.period_days,
        0.0,
        &duration,
    );
    let measured = fixed.map(|candidate| measure(lc, &candidate, raw_sigma));
    let epoch_error = fixed
        .map(|candidate| phase_delta(candidate.t0, signal.epoch_bkjd, signal.period_days).abs());
    AuditRow {
        source_id: signal.source_id,
        period_days: signal.period_days,
        epoch_bkjd: signal.epoch_bkjd,
        duration_hours: signal.duration_hours,
        detected_events: events.len(),
        matched_detected_events: matched_events.len(),
        matched_event_snr_sum: matched_events.values().sum(),
        matched_event_snr_max: matched_events.values().copied().reduce(f64::max),
        matched_event_local_snr_max: matched_local_snr_max,
        event_brightening_snr_max: sparse_event_brightening_control(lc, cfg),
        proposals: proposals.len(),
        exact_proposal_rank,
        closest_exact_proposal_rank: closest_exact.map(|item| item.1),
        closest_exact_proposal_error_fraction: closest_exact.map(|item| item.0),
        harmonic_proposal_rank: harmonic_match.map(|item| item.0),
        harmonic_proposal: harmonic_match.map(|item| item.1),
        closest_harmonic_proposal_rank: closest_harmonic.map(|item| item.1),
        closest_harmonic_proposal: closest_harmonic.map(|item| item.2),
        closest_harmonic_proposal_error_fraction: closest_harmonic.map(|item| item.0),
        center_fold_match_rank: center_fold_match.map(|item| item.0 + 1),
        center_fold_match_mes: center_fold_match.map(|item| item.1.search_mes),
        refined_fold_match_rank: refined_fold_match.map(|item| item.0 + 1),
        refined_fold_match_mes: refined_fold_match.map(|item| item.1.search_mes),
        fixed_period_search_mes: fixed.map(|candidate| candidate.search_mes),
        fixed_period_measured_mes: measured.map(|metrics| metrics.mes),
        fixed_period_epoch_error_days: epoch_error,
    }
}

fn run_audit(args: AuditArgs) -> Result<()> {
    let cfg = load_config(&args.config)?;
    let inputs = scan_inputs(&args.input_dir)?;
    let paths: HashMap<u64, PathBuf> = inputs
        .into_iter()
        .filter(|(_, header)| header.photometry == Photometry::PsfFlat)
        .map(|(path, header)| (header.source_id, path))
        .collect();
    let mut reader = csv::Reader::from_path(&args.signals)
        .with_context(|| format!("reading {}", args.signals.display()))?;
    let signals: Vec<AuditSignal> = reader
        .deserialize()
        .collect::<std::result::Result<_, _>>()?;
    let rows: Result<Vec<AuditRow>> = signals
        .par_iter()
        .map(|signal| {
            let path = paths
                .get(&signal.source_id)
                .with_context(|| format!("missing input for source {}", signal.source_id))?;
            let lc = read_light_curve(path)?;
            Ok(audit_signal(&lc, &cfg, signal))
        })
        .collect();
    let mut rows = rows?;
    rows.sort_by(|left, right| {
        (left.source_id, left.period_days)
            .partial_cmp(&(right.source_id, right.period_days))
            .unwrap_or(Ordering::Equal)
    });
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = csv::Writer::from_path(&args.output)
        .with_context(|| format!("creating {}", args.output.display()))?;
    for row in rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

fn prepare_null_curve(lc: LightCurve, cfg: &Config) -> NullCurve {
    let search = whiten_for_search(&lc, &cfg.search_whitening);
    NullCurve {
        search_sigma: robust_sigma(&search.flux),
        raw_sigma: robust_sigma(&lc.flux),
        photometry: lc,
        search,
    }
}

fn null_mes_at_period(curve: &NullCurve, cfg: &Config, signal: &AuditSignal) -> Option<f64> {
    let duration = [signal.duration_hours / 24.0];
    let candidate = box_search_at_period_with_durations(
        &curve.search,
        cfg,
        curve.search_sigma,
        signal.period_days,
        0.0,
        &duration,
    )?;
    Some(measure(&curve.photometry, &candidate, curve.raw_sigma).mes)
}

fn audit_null_signal(curves: &[NullCurve; 5], cfg: &Config, signal: &AuditSignal) -> NullAuditRow {
    let scrambled: Vec<Option<f64>> = curves[1..]
        .iter()
        .map(|curve| null_mes_at_period(curve, cfg, signal))
        .collect();
    NullAuditRow {
        source_id: signal.source_id,
        period_days: signal.period_days,
        epoch_bkjd: signal.epoch_bkjd,
        duration_hours: signal.duration_hours,
        inverted_mes: null_mes_at_period(&curves[0], cfg, signal),
        scramble_mes_0: scrambled[0],
        scramble_mes_1: scrambled[1],
        scramble_mes_2: scrambled[2],
        scramble_mes_3: scrambled[3],
    }
}

fn prepare_null_curves(lc: &LightCurve, cfg: &Config) -> [NullCurve; 5] {
    let mut inverted = lc.clone();
    for flux in &mut inverted.flux {
        *flux = -*flux;
    }
    [
        prepare_null_curve(inverted, cfg),
        prepare_null_curve(rotated_quarter_control(lc, 0), cfg),
        prepare_null_curve(rotated_quarter_control(lc, 1), cfg),
        prepare_null_curve(rotated_quarter_control(lc, 2), cfg),
        prepare_null_curve(rotated_quarter_control(lc, 3), cfg),
    ]
}

fn run_audit_nulls(args: AuditArgs) -> Result<()> {
    let cfg = load_config(&args.config)?;
    if cfg.control_trials != 4 {
        bail!("null audit requires four configured scramble trials");
    }
    let mut paths = HashMap::new();
    for (path, header) in scan_inputs(&args.input_dir)? {
        if header.photometry == Photometry::PsfFlat
            && paths.insert(header.source_id, path).is_some()
        {
            bail!(
                "duplicate primary-view input for source {}",
                header.source_id
            );
        }
    }
    let mut reader = csv::Reader::from_path(&args.signals)
        .with_context(|| format!("reading {}", args.signals.display()))?;
    let signals: Vec<AuditSignal> = reader
        .deserialize()
        .collect::<std::result::Result<_, _>>()?;
    if signals.is_empty() {
        bail!("null audit requires at least one signal");
    }
    let mut by_source: HashMap<u64, Vec<AuditSignal>> = HashMap::new();
    for signal in signals {
        if !signal.period_days.is_finite()
            || signal.period_days <= 0.0
            || !signal.epoch_bkjd.is_finite()
            || !signal.duration_hours.is_finite()
            || signal.duration_hours <= 0.0
        {
            bail!(
                "invalid null-audit ephemeris for source {}",
                signal.source_id
            );
        }
        by_source.entry(signal.source_id).or_default().push(signal);
    }
    let batches: Result<Vec<Vec<NullAuditRow>>> = by_source
        .into_par_iter()
        .map(|(source_id, signals)| {
            let path = paths
                .get(&source_id)
                .with_context(|| format!("missing input for source {source_id}"))?;
            let lc = read_light_curve(path)?;
            let curves = prepare_null_curves(&lc, &cfg);
            Ok(signals
                .iter()
                .map(|signal| audit_null_signal(&curves, &cfg, signal))
                .collect())
        })
        .collect();
    let mut rows: Vec<NullAuditRow> = batches?.into_iter().flatten().collect();
    rows.sort_by(|left, right| {
        (left.source_id, left.period_days)
            .partial_cmp(&(right.source_id, right.period_days))
            .unwrap_or(Ordering::Equal)
    });
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = csv::Writer::from_path(&args.output)
        .with_context(|| format!("creating {}", args.output.display()))?;
    for row in rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

fn scan_inputs(folder: &Path) -> Result<Vec<(PathBuf, CurveHeader)>> {
    let mut paths: Vec<PathBuf> = fs::read_dir(folder)
        .with_context(|| format!("reading {}", folder.display()))?
        .map(|entry| entry.map(|item| item.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.retain(|path| path.extension().is_some_and(|ext| ext == "fsc"));
    paths.sort();
    if paths.is_empty() {
        bail!("no .fsc light curves found in {}", folder.display());
    }
    let mut identities = HashSet::new();
    let mut inputs = Vec::with_capacity(paths.len());
    for path in paths {
        let mut file = File::open(&path)?;
        let bytes = file.metadata()?.len();
        let header = read_curve_header(&mut file, bytes)
            .with_context(|| format!("invalid input {}", path.display()))?;
        if !identities.insert((header.source_id, header.photometry)) {
            bail!(
                "duplicate source/view input: {} {:?}",
                header.source_id,
                header.photometry
            );
        }
        inputs.push((path, header));
    }
    Ok(inputs)
}

fn reconcile_rows(inputs: &[(PathBuf, CurveHeader)], rows: &[SignalRow]) -> Result<()> {
    let expected: HashSet<_> = inputs
        .iter()
        .map(|(_, header)| (header.source_id, header.photometry))
        .collect();
    let mut found = HashSet::new();
    let mut ranks = HashSet::new();
    for row in rows {
        let identity = (row.source_id, row.photometry);
        if !expected.contains(&identity)
            || !ranks.insert((row.source_id, row.photometry, row.signal_rank))
        {
            bail!("unknown or duplicate source/view/rank in detector output");
        }
        found.insert(identity);
    }
    if found != expected {
        bail!("detector output omitted an input source/view");
    }
    Ok(())
}

fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_curve_header(reader: &mut impl Read, bytes: u64) -> Result<CurveHeader> {
    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        bail!("invalid FSC_LC2 magic");
    }
    let source_id = read_u64(reader)?;
    let photometry = Photometry::from_code(read_u64(reader)?)?;
    let records = read_u64(reader)?;
    let expected = records
        .checked_mul(RECORD_BYTES)
        .and_then(|payload| payload.checked_add(HEADER_BYTES))
        .context("FSC_LC2 record count overflows file length")?;
    if records < 100 || bytes != expected {
        bail!("FSC_LC2 needs >=100 records and exact header/payload length");
    }
    Ok(CurveHeader {
        source_id,
        photometry,
        records: usize::try_from(records)?,
    })
}

fn read_light_curve(path: &Path) -> Result<LightCurve> {
    let file = File::open(path)?;
    let bytes = file.metadata()?.len();
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let header = read_curve_header(&mut reader, bytes)
        .with_context(|| format!("invalid input {}", path.display()))?;
    let n = header.records;
    let mut time = Vec::with_capacity(n);
    let mut flux = Vec::with_capacity(n);
    let mut error = Vec::with_capacity(n);
    let mut quarter = Vec::with_capacity(n);
    let mut cadence = Vec::with_capacity(n);
    let mut record = [0_u8; 22];
    for _ in 0..n {
        reader.read_exact(&mut record)?;
        time.push(f64::from_le_bytes(record[0..8].try_into().unwrap()));
        flux.push(f32::from_le_bytes(record[8..12].try_into().unwrap()) as f64);
        error.push(f32::from_le_bytes(record[12..16].try_into().unwrap()) as f64);
        quarter.push(i16::from_le_bytes(record[16..18].try_into().unwrap()));
        cadence.push(i32::from_le_bytes(record[18..22].try_into().unwrap()));
    }
    if time.iter().chain(&flux).any(|value| !value.is_finite())
        || error
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || time.windows(2).any(|pair| pair[1] < pair[0])
    {
        bail!("{} has invalid or unsorted photometry", path.display());
    }
    Ok(LightCurve {
        source_id: header.source_id,
        photometry: header.photometry,
        time,
        flux,
        error,
        quarter,
        cadence,
    })
}

fn median(values: &[f64]) -> f64 {
    let mut copy: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    median_in_place(&mut copy)
}

fn median_in_place(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let middle = values.len() / 2;
    let even = values.len().is_multiple_of(2);
    let (lower, upper, _) =
        values.select_nth_unstable_by(middle, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    if even {
        let lower_middle = lower
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap();
        (lower_middle + *upper) * 0.5
    } else {
        *upper
    }
}

fn robust_sigma(values: &[f64]) -> f64 {
    let med = median(values);
    let deviations: Vec<f64> = values.iter().map(|value| (value - med).abs()).collect();
    1.4826 * median(&deviations)
}

fn whiten_for_search(lc: &LightCurve, config: &SearchWhiteningConfig) -> LightCurve {
    if !config.enabled || config.window_points < 3 {
        return lc.clone();
    }
    let mut result = lc.clone();
    let requested = if config.window_points.is_multiple_of(2) {
        config.window_points - 1
    } else {
        config.window_points
    };
    let mut start = 0;
    while start < lc.flux.len() {
        let quarter = lc.quarter[start];
        let mut stop = start + 1;
        while stop < lc.flux.len() && lc.quarter[stop] == quarter {
            stop += 1;
        }
        let maximum = (stop - start) | 1;
        let window = requested.min(maximum);
        let half = window / 2;
        let mut local = Vec::with_capacity(window);
        for index in start..stop {
            local.clear();
            for offset in 0..window {
                let position = index as isize + offset as isize - half as isize;
                let position = position.clamp(start as isize, stop as isize - 1) as usize;
                local.push(lc.flux[position]);
            }
            result.flux[index] = lc.flux[index] - median_in_place(&mut local);
        }
        if config.scale_by_quarter {
            let scale = robust_sigma(&result.flux[start..stop]);
            if scale.is_finite() && scale > 0.0 {
                for index in start..stop {
                    result.flux[index] /= scale;
                    result.error[index] /= scale;
                }
            }
        }
        start = stop;
    }
    result
}

fn typical_cadence(lc: &LightCurve) -> f64 {
    let differences: Vec<f64> = lc
        .time
        .windows(2)
        .zip(lc.cadence.windows(2))
        .filter_map(|(times, cadences)| {
            let dt = times[1] - times[0];
            (cadences[1] - cadences[0] == 1 && dt > 0.0 && dt < 0.1).then_some(dt)
        })
        .collect();
    median(&differences)
}

fn select_events(mut candidates: Vec<Event>, max_events: usize) -> Vec<Event> {
    candidates.sort_unstable_by(|a, b| b.snr.partial_cmp(&a.snr).unwrap_or(Ordering::Equal));
    let mut selected: Vec<Event> = Vec::with_capacity(max_events);
    'candidate: for event in candidates {
        for kept in &selected {
            let exclusion = 0.6 * event.duration.max(kept.duration);
            if (event.time - kept.time).abs() < exclusion {
                continue 'candidate;
            }
        }
        selected.push(event);
        if selected.len() == max_events {
            break;
        }
    }
    selected.sort_unstable_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(Ordering::Equal));
    selected
}

fn detect_events(
    lc: &LightCurve,
    cfg: &Config,
    sigma: f64,
    requested_duration_hours: &[f64],
) -> Vec<Event> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Vec::new();
    }
    let n = lc.time.len();
    let cadence = typical_cadence(lc).max(1.0 / 48.94);
    let mut prefix = vec![0.0; n + 1];
    for (index, value) in lc.flux.iter().enumerate() {
        prefix[index + 1] = prefix[index] + value;
    }
    let mut candidates = Vec::new();
    for duration_hours in requested_duration_hours {
        let duration = duration_hours / 24.0;
        let half = duration * 0.5;
        let minimum_points = ((duration / cadence) * 0.45).floor().max(1.0) as usize;
        let mut scores = vec![f64::NEG_INFINITY; n];
        let mut left = 0_usize;
        let mut right = 0_usize;
        for (center, score) in scores.iter_mut().enumerate() {
            while left < center && lc.time[center] - lc.time[left] > half {
                left += 1;
            }
            right = right.max(center + 1);
            while right < n && lc.time[right] - lc.time[center] <= half {
                right += 1;
            }
            let count = right - left;
            if count >= minimum_points {
                let mean = (prefix[right] - prefix[left]) / count as f64;
                *score = -mean * (count as f64).sqrt() / sigma;
            }
        }
        for (center, &score) in scores.iter().enumerate() {
            let previous = center
                .checked_sub(1)
                .map_or(f64::NEG_INFINITY, |i| scores[i]);
            let next = scores.get(center + 1).copied().unwrap_or(f64::NEG_INFINITY);
            if score >= cfg.event_snr_floor && score >= previous && score >= next {
                candidates.push(Event {
                    time: lc.time[center],
                    duration,
                    snr: score,
                });
            }
        }
    }
    select_events(candidates, cfg.max_events)
}

fn measure_single_event(lc: &LightCurve, event: &Event, sigma: f64) -> Option<SingleEventMetrics> {
    let nearest = lc.time.partition_point(|time| *time < event.time);
    let right = nearest.min(lc.time.len() - 1);
    let left = nearest.saturating_sub(1);
    let center = if (lc.time[left] - event.time).abs() <= (lc.time[right] - event.time).abs() {
        left
    } else {
        right
    };
    let quarter = lc.quarter[center];
    let mut inside = Vec::new();
    let mut before = Vec::new();
    let mut after = Vec::new();
    for index in 0..lc.time.len() {
        if lc.quarter[index] != quarter {
            continue;
        }
        let distance = (lc.time[index] - event.time).abs();
        if distance <= event.duration * 0.5 {
            inside.push(lc.flux[index]);
        } else if distance >= event.duration && distance <= event.duration * 4.0 {
            if lc.time[index] < event.time {
                before.push(lc.flux[index]);
            } else {
                after.push(lc.flux[index]);
            }
        }
    }
    if inside.is_empty()
        || before.is_empty()
        || after.is_empty()
        || !sigma.is_finite()
        || sigma <= 0.0
    {
        return None;
    }
    let in_mean = inside.iter().sum::<f64>() / inside.len() as f64;
    let before_mean = before.iter().sum::<f64>() / before.len() as f64;
    let after_mean = after.iter().sum::<f64>() / after.len() as f64;
    let depth = (before_mean - in_mean).min(after_mean - in_mean);
    let baseline_points = before.len().min(after.len());
    let snr = depth / sigma
        * ((inside.len() * baseline_points) as f64 / (inside.len() + baseline_points) as f64)
            .sqrt();
    (depth.is_finite() && snr.is_finite()).then_some(SingleEventMetrics {
        depth_ppm: depth * 1e6,
        snr,
        n_in_transit_points: inside.len(),
    })
}

#[derive(Default)]
struct ProposalBin {
    weight: f64,
    precision_weighted_period: f64,
    precision_weight: f64,
}

fn pair_period_precision_weight(left: &Event, right: &Event, harmonic: usize) -> f64 {
    let timing_variance =
        (left.duration / left.snr.max(0.1)).powi(2) + (right.duration / right.snr.max(0.1)).powi(2);
    (harmonic as f64).powi(2) / timing_variance.max(f64::EPSILON)
}

fn event_coherence(period: f64, events: &[Event]) -> (f64, f64) {
    let mut best_score = 0.0;
    let mut best_period = period;
    let mut anchor_candidates: Vec<&Event> = events
        .iter()
        .filter(|event| event.snr.is_finite())
        .collect();
    anchor_candidates.sort_unstable_by(|a, b| b.snr.partial_cmp(&a.snr).unwrap_or(Ordering::Equal));
    anchor_candidates.truncate(32);
    for anchor in anchor_candidates {
        let mut matched: Vec<(f64, f64, f64)> = Vec::new();
        let mut score = 0.0;
        for event in events {
            let epoch = ((event.time - anchor.time) / period).round();
            let residual = (event.time - anchor.time - epoch * period).abs();
            let tolerance = 0.75 * event.duration.max(anchor.duration).max(1.0 / 24.0);
            if residual <= tolerance {
                let weight = event.snr.max(0.1).powi(2);
                matched.push((epoch, event.time, weight));
                score += event.snr.max(0.0);
            }
        }
        if matched.len() < 3 || score <= best_score {
            continue;
        }
        let weight_sum: f64 = matched.iter().map(|item| item.2).sum();
        let mean_epoch: f64 = matched.iter().map(|item| item.0 * item.2).sum::<f64>() / weight_sum;
        let mean_time: f64 = matched.iter().map(|item| item.1 * item.2).sum::<f64>() / weight_sum;
        let covariance: f64 = matched
            .iter()
            .map(|item| item.2 * (item.0 - mean_epoch) * (item.1 - mean_time))
            .sum();
        let variance: f64 = matched
            .iter()
            .map(|item| item.2 * (item.0 - mean_epoch).powi(2))
            .sum();
        let fitted_period = if variance > 0.0 {
            covariance / variance
        } else {
            period
        };
        if fitted_period.is_finite() && (fitted_period / period - 1.0).abs() < 0.02 {
            best_period = fitted_period;
        }
        best_score = score;
    }
    (best_score, best_period)
}

fn select_period_range(
    proposals: &[Proposal],
    period_min_days: f64,
    period_max_days: f64,
    limit: usize,
) -> Vec<Proposal> {
    if limit == 0 || period_max_days <= period_min_days {
        return Vec::new();
    }
    let bucket_count = 16_usize.min(limit);
    let quota = limit.div_ceil(bucket_count);
    let log_min = period_min_days.ln();
    let log_span = period_max_days.ln() - log_min;
    let mut buckets: Vec<Vec<Proposal>> = vec![Vec::new(); bucket_count];
    for &proposal in proposals {
        if proposal.period < period_min_days || proposal.period > period_max_days {
            continue;
        }
        let fraction = ((proposal.period.ln() - log_min) / log_span).clamp(0.0, 0.999_999);
        let bucket = (fraction * bucket_count as f64) as usize;
        buckets[bucket].push(proposal);
    }
    let mut selected = Vec::with_capacity(limit);
    for bucket in &mut buckets {
        bucket.sort_unstable_by(|a, b| {
            b.event_score
                .partial_cmp(&a.event_score)
                .unwrap_or(Ordering::Equal)
        });
        selected.extend(bucket.iter().take(quota).copied());
    }
    selected.sort_unstable_by(|a, b| {
        b.event_score
            .partial_cmp(&a.event_score)
            .unwrap_or(Ordering::Equal)
    });
    selected.truncate(limit);
    selected
}

fn propose_periods(events: &[Event], cfg: &Config, period_max_days: f64) -> Vec<Proposal> {
    if events.len() < 2 || period_max_days <= cfg.period_min_days {
        return Vec::new();
    }
    let mut bins: HashMap<i64, ProposalBin> = HashMap::new();
    for left in 0..events.len() {
        for right in (left + 1)..events.len() {
            let dt = events[right].time - events[left].time;
            let first_harmonic = (dt / period_max_days).ceil().max(1.0) as usize;
            let last_harmonic = (dt / cfg.period_min_days).floor() as usize;
            for harmonic in first_harmonic..=last_harmonic.min(cfg.max_pair_harmonic) {
                let period = dt / harmonic as f64;
                if period < cfg.period_min_days || period > period_max_days {
                    continue;
                }
                let key = (period.ln() / cfg.period_log_bin_width).round() as i64;
                let weight =
                    (events[left].snr * events[right].snr).sqrt() / (harmonic as f64).sqrt();
                let precision_weight =
                    pair_period_precision_weight(&events[left], &events[right], harmonic);
                let bin = bins.entry(key).or_default();
                bin.weight += weight;
                bin.precision_weighted_period += period * precision_weight;
                bin.precision_weight += precision_weight;
            }
        }
    }
    let proposals: Vec<Proposal> = bins
        .into_values()
        .filter(|bin| bin.weight > 0.0)
        .map(|bin| {
            let period = bin.precision_weighted_period / bin.precision_weight;
            let (coherence, _) = event_coherence(period, events);
            Proposal {
                period,
                event_score: coherence + 0.02 * bin.weight,
            }
        })
        .filter(|proposal| proposal.event_score > 0.0)
        .collect();
    // Keep core candidates through box-likelihood scoring to protect them from dense subharmonic pools.
    let mut selected = match (cfg.priority_period_min_days, cfg.priority_period_max_days) {
        (Some(priority_min), Some(priority_max)) => {
            let mut core = select_period_range(
                &proposals,
                priority_min,
                priority_max.min(period_max_days),
                cfg.max_period_proposals,
            );
            let mut short = select_period_range(
                &proposals,
                cfg.period_min_days,
                priority_min,
                cfg.max_extension_period_proposals,
            );
            short.retain(|proposal| proposal.period < priority_min);
            let mut long = select_period_range(
                &proposals,
                priority_max,
                period_max_days,
                cfg.max_extension_period_proposals,
            );
            long.retain(|proposal| proposal.period > priority_max);
            core.extend(short);
            core.extend(long);
            core
        }
        _ => select_period_range(
            &proposals,
            cfg.period_min_days,
            period_max_days,
            cfg.max_period_proposals,
        ),
    };
    selected.sort_unstable_by(|a, b| {
        b.event_score
            .partial_cmp(&a.event_score)
            .unwrap_or(Ordering::Equal)
    });
    selected
}

fn periodic_max_days(lc: &LightCurve, cfg: &Config) -> f64 {
    let baseline = lc.time.last().unwrap() - lc.time.first().unwrap();
    let epoch_intervals = cfg.gates.n_transits_min.saturating_sub(1).max(1) as f64;
    cfg.period_max_days.min(baseline / epoch_intervals)
}

fn priority_duration_hours(cfg: &Config) -> Vec<f64> {
    cfg.priority_duration_max_hours.map_or_else(
        || cfg.duration_hours.clone(),
        |maximum| {
            cfg.duration_hours
                .iter()
                .copied()
                .filter(|duration| *duration <= maximum)
                .collect()
        },
    )
}

fn period_duration_days(cfg: &Config, period: f64) -> Vec<f64> {
    let priority = cfg
        .priority_period_min_days
        .zip(cfg.priority_period_max_days)
        .is_some_and(|(minimum, maximum)| period >= minimum && period <= maximum);
    let maximum = priority
        .then_some(cfg.priority_duration_max_hours)
        .flatten();
    cfg.duration_hours
        .iter()
        .copied()
        .filter(|duration| maximum.is_none_or(|value| *duration <= value))
        .map(|duration| duration / 24.0)
        .collect()
}

fn box_search_at_period_with_durations(
    lc: &LightCurve,
    cfg: &Config,
    sigma: f64,
    period: f64,
    event_score: f64,
    durations: &[f64],
) -> Option<BoxCandidate> {
    let shortest = cfg.duration_hours[0] / 24.0;
    let bins_for_resolution =
        ((2.0 * period / shortest).ceil() as usize).clamp(cfg.phase_bins_min, cfg.phase_bins_max);
    let bins = bins_for_resolution;
    let t_ref = lc.time[0];
    let mut counts = vec![0_u32; bins];
    let mut sums = vec![0.0_f64; bins];
    for (&time, &flux) in lc.time.iter().zip(&lc.flux) {
        let phase = ((time - t_ref) / period).rem_euclid(1.0);
        let bin = ((phase * bins as f64).floor() as usize).min(bins - 1);
        counts[bin] += 1;
        sums[bin] += flux;
    }
    let total_count: usize = counts.iter().map(|&value| value as usize).sum();
    let total_sum: f64 = sums.iter().sum();
    if total_count < cfg.gates.n_in_transit_points_min * 2 {
        return None;
    }
    let mut prefix_count = vec![0_usize; 2 * bins + 1];
    let mut prefix_sum = vec![0.0_f64; 2 * bins + 1];
    for index in 0..(2 * bins) {
        let wrapped = index % bins;
        prefix_count[index + 1] = prefix_count[index] + counts[wrapped] as usize;
        prefix_sum[index + 1] = prefix_sum[index] + sums[wrapped];
    }

    let mut best: Option<BoxCandidate> = None;
    for &requested_duration in durations {
        if requested_duration / period > cfg.gates.duration_fraction_max {
            continue;
        }
        let width =
            ((requested_duration / period * bins as f64).round() as usize).clamp(1, bins / 2);
        let actual_duration = width as f64 / bins as f64 * period;
        for start in 0..bins {
            let in_count = prefix_count[start + width] - prefix_count[start];
            let out_count = total_count.saturating_sub(in_count);
            if in_count < cfg.gates.n_in_transit_points_min || out_count < in_count {
                continue;
            }
            let in_sum = prefix_sum[start + width] - prefix_sum[start];
            let out_sum = total_sum - in_sum;
            let depth = out_sum / out_count as f64 - in_sum / in_count as f64;
            let mes = depth / sigma * ((in_count * out_count) as f64 / total_count as f64).sqrt();
            if !mes.is_finite() || mes <= best.map_or(f64::NEG_INFINITY, |item| item.search_mes) {
                continue;
            }
            let center_phase = ((start as f64 + width as f64 * 0.5) / bins as f64).rem_euclid(1.0);
            best = Some(BoxCandidate {
                period,
                t0: t_ref + center_phase * period,
                duration: actual_duration,
                search_mes: mes,
                event_score,
                search_pass: 1,
            });
        }
    }
    best
}

fn phase_delta(time: f64, t0: f64, period: f64) -> f64 {
    (time - t0 + period * 0.5).rem_euclid(period) - period * 0.5
}

fn sample_std(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    variance.sqrt()
}

fn mean_and_error(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::INFINITY);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let error = if values.len() >= 2 {
        sample_std(values) / (values.len() as f64).sqrt()
    } else {
        f64::INFINITY
    };
    (mean, error)
}

fn red_noise_ratio(lc: &LightCurve, candidate: &BoxCandidate, sigma: f64) -> f64 {
    let cadence = typical_cadence(lc);
    if !cadence.is_finite() || cadence <= 0.0 || sigma <= 0.0 {
        return f64::INFINITY;
    }
    let width = (candidate.duration / cadence).round().max(1.0) as usize;
    let mut means = Vec::new();
    let mut start = 0_usize;
    while start + width <= lc.time.len() {
        let end = start + width;
        let contiguous = lc.quarter[start] == lc.quarter[end - 1]
            && lc.time[end - 1] - lc.time[start] <= candidate.duration * 1.5;
        let outside = (start..end).all(|index| {
            phase_delta(lc.time[index], candidate.t0, candidate.period).abs() > candidate.duration
        });
        if contiguous && outside {
            means.push(lc.flux[start..end].iter().sum::<f64>() / width as f64);
        }
        start += width;
    }
    let binned_sigma = robust_sigma(&means);
    if !binned_sigma.is_finite() {
        f64::INFINITY
    } else {
        binned_sigma / (sigma / (width as f64).sqrt())
    }
}

fn measure(lc: &LightCurve, candidate: &BoxCandidate, sigma: f64) -> Metrics {
    let mut in_flux = Vec::new();
    let mut out_flux = Vec::new();
    let mut shoulder_flux = Vec::new();
    let mut secondary_flux = Vec::new();
    let mut inner_flux = Vec::new();
    let mut edge_flux = Vec::new();
    let mut quarters = HashSet::new();
    let mut transits: HashMap<i64, Vec<f64>> = HashMap::new();
    for index in 0..lc.time.len() {
        let delta = phase_delta(lc.time[index], candidate.t0, candidate.period);
        if delta.abs() <= candidate.duration * 0.5 {
            in_flux.push(lc.flux[index]);
            quarters.insert(lc.quarter[index]);
            let epoch = ((lc.time[index] - candidate.t0) / candidate.period).round() as i64;
            transits.entry(epoch).or_default().push(lc.flux[index]);
            if delta.abs() <= candidate.duration * 0.25 {
                inner_flux.push(lc.flux[index]);
            } else {
                edge_flux.push(lc.flux[index]);
            }
        } else {
            out_flux.push(lc.flux[index]);
            if delta.abs() >= candidate.duration * 0.75 && delta.abs() <= candidate.duration * 2.0 {
                shoulder_flux.push(lc.flux[index]);
            }
        }
        let secondary_delta = phase_delta(
            lc.time[index],
            candidate.t0 + candidate.period * 0.5,
            candidate.period,
        );
        if secondary_delta.abs() <= candidate.duration * 0.5 {
            secondary_flux.push(lc.flux[index]);
        }
    }
    let out_mean = if out_flux.is_empty() {
        0.0
    } else {
        out_flux.iter().sum::<f64>() / out_flux.len() as f64
    };
    let in_mean = if in_flux.is_empty() {
        0.0
    } else {
        in_flux.iter().sum::<f64>() / in_flux.len() as f64
    };
    let depth = out_mean - in_mean;
    let mes = if in_flux.is_empty() || out_flux.is_empty() {
        0.0
    } else {
        depth / sigma * ((in_flux.len() * out_flux.len()) as f64 / lc.time.len() as f64).sqrt()
    };
    let shoulder_mean = if shoulder_flux.is_empty() {
        out_mean
    } else {
        shoulder_flux.iter().sum::<f64>() / shoulder_flux.len() as f64
    };
    let local_depth = shoulder_mean - in_mean;
    let local_mes = if in_flux.is_empty() || shoulder_flux.is_empty() {
        0.0
    } else {
        local_depth / sigma
            * ((in_flux.len() * shoulder_flux.len()) as f64
                / (in_flux.len() + shoulder_flux.len()) as f64)
                .sqrt()
    };
    let local_to_global_depth = if depth > 0.0 {
        local_depth / depth
    } else {
        f64::NEG_INFINITY
    };

    let mut transit_depths: Vec<(i64, f64, usize)> = transits
        .into_iter()
        .map(|(epoch, values)| {
            let depth = out_mean - values.iter().sum::<f64>() / values.len() as f64;
            (epoch, depth, values.len())
        })
        .collect();
    transit_depths.sort_by_key(|item| item.0);
    let strengths: Vec<f64> = transit_depths
        .iter()
        .map(|item| item.1.max(0.0) * (item.2 as f64).sqrt())
        .collect();
    let total_strength: f64 = strengths.iter().sum();
    let single_event_fraction = if total_strength > 0.0 {
        strengths.iter().copied().fold(0.0, f64::max) / total_strength
    } else {
        1.0
    };

    let odd: Vec<f64> = transit_depths
        .iter()
        .filter(|item| item.0.rem_euclid(2) == 1)
        .map(|item| item.1)
        .collect();
    let even: Vec<f64> = transit_depths
        .iter()
        .filter(|item| item.0.rem_euclid(2) == 0)
        .map(|item| item.1)
        .collect();
    let (odd_mean, odd_error) = mean_and_error(&odd);
    let (even_mean, even_error) = mean_and_error(&even);
    let odd_even_sigma = if odd_mean.is_finite() && even_mean.is_finite() {
        (odd_mean - even_mean).abs() / odd_error.hypot(even_error).max(1e-12)
    } else {
        0.0
    };

    let secondary_mean = if secondary_flux.is_empty() {
        out_mean
    } else {
        secondary_flux.iter().sum::<f64>() / secondary_flux.len() as f64
    };
    let secondary_snr = if secondary_flux.is_empty() {
        0.0
    } else {
        ((out_mean - secondary_mean) / sigma * (secondary_flux.len() as f64).sqrt()).max(0.0)
    };
    let inner_depth = if inner_flux.is_empty() {
        depth
    } else {
        out_mean - inner_flux.iter().sum::<f64>() / inner_flux.len() as f64
    };
    let edge_depth = if edge_flux.is_empty() {
        depth
    } else {
        out_mean - edge_flux.iter().sum::<f64>() / edge_flux.len() as f64
    };
    let v_shape_score = if inner_depth > 0.0 {
        ((inner_depth - edge_depth) / inner_depth).clamp(0.0, 2.0)
    } else {
        0.0
    };
    let baseline = lc.time.last().unwrap() - lc.time.first().unwrap();
    Metrics {
        period: candidate.period,
        t0: candidate.t0,
        duration: candidate.duration,
        depth_ppm: depth * 1e6,
        mes,
        local_depth_ppm: local_depth * 1e6,
        local_mes,
        local_to_global_depth,
        n_transits: transit_depths.len(),
        n_in_transit_points: in_flux.len(),
        n_quarters: quarters.len(),
        odd_even_sigma,
        secondary_snr,
        single_event_fraction,
        red_noise_ratio: red_noise_ratio(lc, candidate, sigma),
        v_shape_score,
        robust_sigma_ppm: sigma * 1e6,
        period_resolution_days: candidate.duration * candidate.period / baseline.max(1.0),
    }
}

fn transit_duration_consistency(
    lc: &LightCurve,
    candidate: &BoxCandidate,
    sigma: f64,
    cfg: &Config,
) -> DurationConsistency {
    let test_cfg = &cfg.survivor_validation.duration_consistency;
    if !test_cfg.enabled {
        return DurationConsistency::not_run();
    }
    let maximum = (candidate.duration * 4.0).min(candidate.period * 0.12);
    let minimum = candidate.duration * 0.25;
    let mut durations: Vec<f64> = cfg
        .duration_hours
        .iter()
        .map(|hours| hours / 24.0)
        .filter(|duration| *duration >= minimum && *duration <= maximum)
        .collect();
    durations.push(candidate.duration);
    durations.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    durations.dedup_by(|left, right| (*left / *right - 1.0).abs() < 1e-6);
    let max_duration = durations.iter().copied().fold(candidate.duration, f64::max);
    let window = max_duration * 1.5;
    let baseline_inner = (candidate.duration * 1.25).max(max_duration * 0.6);
    let mut events: HashMap<i64, Vec<(f64, f64)>> = HashMap::new();
    for (&time, &flux) in lc.time.iter().zip(&lc.flux) {
        let delta = phase_delta(time, candidate.t0, candidate.period);
        if delta.abs() <= window {
            let epoch = ((time - candidate.t0) / candidate.period).round() as i64;
            events.entry(epoch).or_default().push((delta, flux));
        }
    }

    let mut fitted_log2_ratios = Vec::new();
    for samples in events.into_values() {
        let baseline: Vec<f64> = samples
            .iter()
            .filter_map(|(delta, flux)| {
                (delta.abs() >= baseline_inner && delta.abs() <= window).then_some(*flux)
            })
            .collect();
        if baseline.len() < 2 {
            continue;
        }
        let baseline_level = median(&baseline);
        let mut best: Option<(f64, f64)> = None;
        for &duration in &durations {
            let inside: Vec<f64> = samples
                .iter()
                .filter_map(|(delta, flux)| (delta.abs() <= duration * 0.5).then_some(*flux))
                .collect();
            if inside.is_empty() {
                continue;
            }
            let depth = baseline_level - inside.iter().sum::<f64>() / inside.len() as f64;
            let snr = depth / sigma.max(1e-12) * (inside.len() as f64).sqrt();
            if snr.is_finite() && snr > best.map_or(f64::NEG_INFINITY, |item| item.1) {
                best = Some((duration, snr));
            }
        }
        if let Some((duration, snr)) = best
            && snr >= test_cfg.event_snr_min
        {
            fitted_log2_ratios.push((duration / candidate.duration).log2());
        }
    }

    if fitted_log2_ratios.len() < test_cfg.min_measurable_events {
        return DurationConsistency {
            status: "NOT_TESTABLE",
            measurable_events: fitted_log2_ratios.len(),
            consistent_fraction: None,
            log2_duration_mad: None,
        };
    }
    let consistent = fitted_log2_ratios
        .iter()
        .filter(|log_ratio| {
            let ratio = 2_f64.powf(**log_ratio);
            ratio >= test_cfg.duration_ratio_min && ratio <= test_cfg.duration_ratio_max
        })
        .count();
    let fraction = consistent as f64 / fitted_log2_ratios.len() as f64;
    let center = median(&fitted_log2_ratios);
    let deviations: Vec<f64> = fitted_log2_ratios
        .iter()
        .map(|value| (value - center).abs())
        .collect();
    DurationConsistency {
        status: if fraction >= test_cfg.min_consistent_fraction {
            "PASS"
        } else {
            "FAIL"
        },
        measurable_events: fitted_log2_ratios.len(),
        consistent_fraction: Some(fraction),
        log2_duration_mad: Some(median(&deviations)),
    }
}

fn subset_by_quarter(lc: &LightCurve, quarter: i16, retain_quarter: bool) -> LightCurve {
    let mut subset = LightCurve {
        source_id: lc.source_id,
        photometry: lc.photometry,
        time: Vec::new(),
        flux: Vec::new(),
        error: Vec::new(),
        quarter: Vec::new(),
        cadence: Vec::new(),
    };
    for index in 0..lc.time.len() {
        if (lc.quarter[index] == quarter) == retain_quarter {
            subset.time.push(lc.time[index]);
            subset.flux.push(lc.flux[index]);
            subset.error.push(lc.error[index]);
            subset.quarter.push(lc.quarter[index]);
            subset.cadence.push(lc.cadence[index]);
        }
    }
    subset
}

fn predicted_depth(lc: &LightCurve, candidate: &BoxCandidate) -> Option<(f64, f64)> {
    let mut inside = Vec::new();
    let mut shoulder = Vec::new();
    for (&time, &flux) in lc.time.iter().zip(&lc.flux) {
        let delta = phase_delta(time, candidate.t0, candidate.period).abs();
        if delta <= candidate.duration * 0.5 {
            inside.push(flux);
        } else if delta >= candidate.duration * 0.75 && delta <= candidate.duration * 2.0 {
            shoulder.push(flux);
        }
    }
    if inside.is_empty() || shoulder.len() < 2 {
        return None;
    }
    let outside = shoulder.iter().sum::<f64>() / shoulder.len() as f64;
    let within = inside.iter().sum::<f64>() / inside.len() as f64;
    let sigma = robust_sigma(&lc.flux);
    if !sigma.is_finite() || sigma <= 0.0 {
        return None;
    }
    let depth = outside - within;
    let snr = depth / sigma
        * ((inside.len() * shoulder.len()) as f64 / (inside.len() + shoulder.len()) as f64).sqrt();
    Some((depth, snr))
}

fn quarter_leave_one_out(lc: &LightCurve, candidate: &BoxCandidate, cfg: &Config) -> QuarterLoo {
    let test_cfg = &cfg.survivor_validation.quarter_loo;
    if !test_cfg.enabled {
        return QuarterLoo::not_run();
    }
    let mut quarters: Vec<i16> = lc
        .quarter
        .iter()
        .copied()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    quarters.sort_unstable();
    let middle = (test_cfg.period_refine_steps / 2) as isize;
    let duration = [candidate.duration];
    let mut testable = 0_usize;
    let mut positive = 0_usize;
    let mut depth_ratios = Vec::new();

    for quarter in quarters {
        let heldout = subset_by_quarter(lc, quarter, true);
        if predicted_depth(&heldout, candidate).is_none() {
            continue;
        }
        testable += 1;
        let training = subset_by_quarter(lc, quarter, false);
        if training.time.len() < cfg.gates.n_in_transit_points_min * 2 {
            continue;
        }
        let sigma = robust_sigma(&training.flux);
        if !sigma.is_finite() || sigma <= 0.0 {
            continue;
        }
        let mut best: Option<BoxCandidate> = None;
        for step in 0..test_cfg.period_refine_steps {
            let offset = step as isize - middle;
            let period = candidate.period
                * (1.0 + offset as f64 * test_cfg.period_refine_fraction / middle.max(1) as f64);
            if let Some(fitted) = box_search_at_period_with_durations(
                &training,
                cfg,
                sigma,
                period,
                candidate.event_score,
                &duration,
            ) {
                let phase_offset = phase_delta(fitted.t0, candidate.t0, candidate.period).abs();
                let tolerance = (candidate.duration * 1.5).max(candidate.period * 0.01);
                if phase_offset <= tolerance
                    && fitted.search_mes > best.map_or(f64::NEG_INFINITY, |item| item.search_mes)
                {
                    best = Some(fitted);
                }
            }
        }
        let Some(fitted) = best else {
            continue;
        };
        let Some((train_depth, _)) = predicted_depth(&training, &fitted) else {
            continue;
        };
        let Some((heldout_depth, heldout_snr)) = predicted_depth(&heldout, &fitted) else {
            continue;
        };
        if train_depth > 0.0 {
            depth_ratios.push(heldout_depth / train_depth);
        }
        if heldout_depth > 0.0 && heldout_snr >= test_cfg.heldout_snr_min {
            positive += 1;
        }
    }

    if testable < test_cfg.min_testable_quarters {
        return QuarterLoo {
            status: "NOT_TESTABLE",
            testable_quarters: testable,
            positive_fraction: None,
            median_depth_ratio: None,
        };
    }
    let positive_fraction = positive as f64 / testable as f64;
    let median_depth_ratio = if depth_ratios.is_empty() {
        None
    } else {
        Some(median(&depth_ratios))
    };
    let passed = positive_fraction >= test_cfg.min_positive_fraction
        && median_depth_ratio.is_some_and(|ratio| ratio >= test_cfg.min_median_depth_ratio);
    QuarterLoo {
        status: if passed { "PASS" } else { "FAIL" },
        testable_quarters: testable,
        positive_fraction: Some(positive_fraction),
        median_depth_ratio,
    }
}

fn non_control_checks(metrics: &Metrics, gates: &Gates) -> bool {
    gates
        .local_mes_min
        .is_none_or(|threshold| metrics.local_mes >= threshold)
        && gates
            .local_to_global_depth_min
            .is_none_or(|threshold| metrics.local_to_global_depth >= threshold)
        && gates
            .v_shape_score_max
            .is_none_or(|threshold| metrics.v_shape_score <= threshold)
}

fn eb_reason(metrics: &Metrics, gates: &Gates) -> bool {
    metrics.depth_ppm > gates.depth_ppm_max
        || metrics.odd_even_sigma > gates.odd_even_sigma_max
        || metrics.secondary_snr > gates.secondary_snr_max
}

fn fixed_systematic_reason(metrics: &Metrics, gates: &Gates) -> bool {
    metrics.n_quarters < gates.n_quarters_min
        || metrics.single_event_fraction > gates.single_event_fraction_max
        || metrics.red_noise_ratio > gates.red_noise_ratio_max
        || !non_control_checks(metrics, gates)
}

fn needs_controls_for_classification(metrics: &Metrics, gates: &Gates) -> bool {
    let controls_enabled =
        gates.dip_brightening_mes_ratio_min.is_some() || gates.dip_control_mes_excess_min.is_some();
    controls_enabled
        && metrics.mes >= gates.mes_min
        && !eb_reason(metrics, gates)
        && !fixed_systematic_reason(metrics, gates)
}

fn classify(
    metrics: &Metrics,
    gates: &Gates,
    brightening_max_mes: f64,
    phase_scramble_max_mes: f64,
) -> &'static str {
    let brightening_ratio = metrics.mes / brightening_max_mes.max(1e-9);
    let control_excess = metrics.mes - brightening_max_mes.max(phase_scramble_max_mes);
    let enhanced_checks = non_control_checks(metrics, gates)
        && gates
            .dip_brightening_mes_ratio_min
            .is_none_or(|threshold| brightening_ratio >= threshold)
        && gates
            .dip_control_mes_excess_min
            .is_none_or(|threshold| control_excess >= threshold);
    let transit_like = metrics.mes >= gates.mes_min
        && metrics.n_transits >= gates.n_transits_min
        && metrics.n_in_transit_points >= gates.n_in_transit_points_min
        && metrics.n_quarters >= gates.n_quarters_min
        && metrics.depth_ppm >= gates.depth_ppm_min
        && metrics.depth_ppm <= gates.depth_ppm_max
        && metrics.odd_even_sigma <= gates.odd_even_sigma_max
        && metrics.secondary_snr <= gates.secondary_snr_max
        && metrics.single_event_fraction <= gates.single_event_fraction_max
        && metrics.red_noise_ratio <= gates.red_noise_ratio_max
        && metrics.duration / metrics.period <= gates.duration_fraction_max
        && enhanced_checks;
    if transit_like {
        "TRANSIT_LIKE"
    } else if metrics.mes >= gates.mes_min && eb_reason(metrics, gates) {
        "EB_LIKE"
    } else if metrics.mes >= gates.mes_min
        && (fixed_systematic_reason(metrics, gates) || !enhanced_checks)
    {
        "SYSTEMATIC_LIKE"
    } else {
        "LOW_SIGNIFICANCE"
    }
}

fn harmonic_related(period: f64, reference: f64) -> bool {
    let ratio = period / reference;
    (1..=8).any(|harmonic| {
        let harmonic = harmonic as f64;
        (ratio / harmonic - 1.0).abs() < 0.01 || (ratio * harmonic - 1.0).abs() < 0.01
    })
}

fn select_candidates_where(
    candidates: &[BoxCandidate],
    limit: usize,
    include: impl Fn(f64) -> bool,
) -> Vec<BoxCandidate> {
    let mut selected: Vec<BoxCandidate> = Vec::with_capacity(limit);
    for &candidate in candidates {
        if !include(candidate.period)
            || selected
                .iter()
                .any(|kept| (candidate.period / kept.period - 1.0).abs() < 0.002)
        {
            continue;
        }
        selected.push(candidate);
        if selected.len() == limit {
            break;
        }
    }
    selected
}

fn periodic_candidates(
    search_curve: &LightCurve,
    cfg: &Config,
    search_sigma: f64,
    include_sparse_durations: bool,
) -> (Vec<BoxCandidate>, Vec<Event>) {
    let proposal_duration_hours = priority_duration_hours(cfg);
    let proposal_events = detect_events(search_curve, cfg, search_sigma, &proposal_duration_hours);
    let events = if !include_sparse_durations || proposal_duration_hours == cfg.duration_hours {
        proposal_events.clone()
    } else {
        let priority_maximum = cfg.priority_duration_max_hours.unwrap();
        let extension_duration_hours: Vec<f64> = cfg
            .duration_hours
            .iter()
            .copied()
            .filter(|duration| *duration > priority_maximum)
            .collect();
        let mut combined = proposal_events.clone();
        combined.extend(detect_events(
            search_curve,
            cfg,
            search_sigma,
            &extension_duration_hours,
        ));
        select_events(combined, cfg.max_events)
    };
    let baseline = search_curve.time.last().unwrap() - search_curve.time.first().unwrap();
    let period_max_days = periodic_max_days(search_curve, cfg);
    let proposals = propose_periods(&proposal_events, cfg, period_max_days);
    let shortest = cfg.duration_hours[0] / 24.0;
    let mut candidates = Vec::new();
    for proposal in proposals {
        let timing_step = (shortest * proposal.period / baseline.max(1.0))
            .max(proposal.period * cfg.period_refine_fraction / cfg.period_refine_steps as f64);
        let middle = (cfg.period_refine_steps / 2) as isize;
        for step in 0..cfg.period_refine_steps {
            let offset = step as isize - middle;
            let period = proposal.period + offset as f64 * timing_step;
            if period < cfg.period_min_days || period > period_max_days {
                continue;
            }
            let durations = period_duration_days(cfg, period);
            if let Some(candidate) = box_search_at_period_with_durations(
                search_curve,
                cfg,
                search_sigma,
                period,
                proposal.event_score,
                &durations,
            ) {
                candidates.push(candidate);
            }
        }
    }
    candidates.sort_unstable_by(|a, b| {
        b.search_mes
            .partial_cmp(&a.search_mes)
            .unwrap_or(Ordering::Equal)
    });
    let mut selected = match (cfg.priority_period_min_days, cfg.priority_period_max_days) {
        (Some(priority_min), Some(priority_max)) => {
            let mut core =
                select_candidates_where(&candidates, cfg.max_signals_per_target, |period| {
                    period >= priority_min && period <= priority_max
                });
            core.extend(select_candidates_where(
                &candidates,
                cfg.max_extension_signals_per_target,
                |period| period < priority_min,
            ));
            core.extend(select_candidates_where(
                &candidates,
                cfg.max_extension_signals_per_target,
                |period| period > priority_max,
            ));
            core
        }
        _ => select_candidates_where(&candidates, cfg.max_signals_per_target, |_| true),
    };
    selected.sort_unstable_by(|a, b| {
        b.search_mes
            .partial_cmp(&a.search_mes)
            .unwrap_or(Ordering::Equal)
    });
    (selected, events)
}

fn eligible_residual_seed(metrics: &Metrics, gates: &Gates) -> bool {
    metrics.mes >= gates.mes_min
        && metrics.n_transits >= gates.n_transits_min
        && metrics.n_in_transit_points >= gates.n_in_transit_points_min
        && metrics.n_quarters >= gates.n_quarters_min
        && metrics.depth_ppm >= gates.depth_ppm_min
        && metrics.depth_ppm <= gates.depth_ppm_max
        && metrics.duration / metrics.period <= gates.duration_fraction_max
        && !eb_reason(metrics, gates)
        && !fixed_systematic_reason(metrics, gates)
}

fn mask_candidate(lc: &LightCurve, candidate: &BoxCandidate) -> LightCurve {
    let keep: Vec<usize> = lc
        .time
        .iter()
        .enumerate()
        .filter_map(|(index, time)| {
            (phase_delta(*time, candidate.t0, candidate.period).abs() > candidate.duration)
                .then_some(index)
        })
        .collect();
    LightCurve {
        source_id: lc.source_id,
        photometry: lc.photometry,
        time: keep.iter().map(|index| lc.time[*index]).collect(),
        flux: keep.iter().map(|index| lc.flux[*index]).collect(),
        error: keep.iter().map(|index| lc.error[*index]).collect(),
        quarter: keep.iter().map(|index| lc.quarter[*index]).collect(),
        cadence: keep.iter().map(|index| lc.cadence[*index]).collect(),
    }
}

fn extend_residual_candidates(
    lc: &LightCurve,
    search_curve: &LightCurve,
    cfg: &Config,
    sigma: f64,
    measured: &mut Vec<(BoxCandidate, Metrics)>,
) {
    let Some((seed, _)) = measured
        .iter()
        .find(|(_, metrics)| eligible_residual_seed(metrics, &cfg.gates))
    else {
        return;
    };
    let residual_curve = mask_candidate(search_curve, seed);
    let residual_sigma = robust_sigma(&residual_curve.flux);
    let (mut residual, _) = periodic_candidates(&residual_curve, cfg, residual_sigma, false);
    residual.retain(|candidate| {
        !measured
            .iter()
            .any(|(kept, _)| (candidate.period / kept.period - 1.0).abs() < 0.002)
    });
    residual.truncate(cfg.max_residual_signals_per_target);
    measured.extend(residual.into_iter().map(|mut candidate| {
        candidate.search_pass = 2;
        let metrics = measure(lc, &candidate, sigma);
        (candidate, metrics)
    }));
    measured.sort_unstable_by(|left, right| {
        right
            .0
            .search_mes
            .partial_cmp(&left.0.search_mes)
            .unwrap_or(Ordering::Equal)
    });
}

fn scan_candidates(
    lc: &LightCurve,
    cfg: &Config,
    allow_residual: bool,
) -> (f64, Vec<(BoxCandidate, Metrics)>, Vec<Event>) {
    let sigma = robust_sigma(&lc.flux);
    let search_curve = whiten_for_search(lc, &cfg.search_whitening);
    let search_sigma = robust_sigma(&search_curve.flux);
    let (selected, events) = periodic_candidates(&search_curve, cfg, search_sigma, true);
    let mut measured: Vec<(BoxCandidate, Metrics)> = selected
        .into_iter()
        .map(|candidate| (candidate, measure(lc, &candidate, sigma)))
        .collect();
    if allow_residual && cfg.max_residual_signals_per_target > 0 {
        extend_residual_candidates(lc, &search_curve, cfg, sigma, &mut measured);
    }
    (sigma, measured, events)
}

fn rotated_quarter_control(lc: &LightCurve, trial: usize) -> LightCurve {
    let mut control = lc.clone();
    let mut by_quarter: HashMap<i16, Vec<usize>> = HashMap::new();
    for (index, &quarter) in lc.quarter.iter().enumerate() {
        by_quarter.entry(quarter).or_default().push(index);
    }
    for (&quarter, indices) in &by_quarter {
        if indices.len() < 2 {
            continue;
        }
        let mut state = lc.source_id
            ^ (trial as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ (quarter as i64 as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        state ^= state >> 30;
        state = state.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        state ^= state >> 27;
        state = state.wrapping_mul(0x94d0_49bb_1331_11eb);
        state ^= state >> 31;
        let shift = (state as usize % (indices.len() - 1)) + 1;
        for (position, &destination) in indices.iter().enumerate() {
            let source = indices[(position + shift) % indices.len()];
            control.flux[destination] = lc.flux[source];
            control.error[destination] = lc.error[source];
        }
    }
    control
}

fn brightening_control(lc: &LightCurve, cfg: &Config, allow_residual: bool) -> f64 {
    let mut inverted = lc.clone();
    for value in &mut inverted.flux {
        *value = -*value;
    }
    let (_, brightening_candidates, _) = scan_candidates(&inverted, cfg, allow_residual);
    brightening_candidates
        .iter()
        .map(|item| item.1.mes)
        .fold(0.0, f64::max)
}

fn phase_scramble_control(lc: &LightCurve, cfg: &Config, allow_residual: bool) -> f64 {
    let mut scramble_max: f64 = 0.0;
    for trial in 0..cfg.control_trials {
        let control = rotated_quarter_control(lc, trial);
        let (_, candidates, _) = scan_candidates(&control, cfg, allow_residual);
        scramble_max =
            scramble_max.max(candidates.iter().map(|item| item.1.mes).fold(0.0, f64::max));
    }
    scramble_max
}

fn maximum_event_mes(lc: &LightCurve, cfg: &Config) -> f64 {
    let search_curve = whiten_for_search(lc, &cfg.search_whitening);
    let search_sigma = robust_sigma(&search_curve.flux);
    let raw_sigma = robust_sigma(&lc.flux);
    detect_events(&search_curve, cfg, search_sigma, &cfg.duration_hours)
        .iter()
        .filter_map(|event| measure_single_event(lc, event, raw_sigma))
        .map(|metrics| metrics.snr)
        .fold(0.0, f64::max)
}

fn sparse_event_brightening_control(lc: &LightCurve, cfg: &Config) -> f64 {
    let mut inverted = lc.clone();
    for value in &mut inverted.flux {
        *value = -*value;
    }
    maximum_event_mes(&inverted, cfg)
}

fn no_signal_row(lc: &LightCurve, cfg: &Config, baseline: f64) -> SignalRow {
    SignalRow {
        protocol_version: cfg.protocol_version.clone(),
        source_id: lc.source_id,
        photometry: lc.photometry,
        signal_rank: 0,
        classification: "NO_SIGNAL".to_string(),
        search_pass: None,
        period_days: None,
        t0_bkjd: None,
        duration_hours: None,
        depth_ppm: None,
        search_mes: None,
        mes: None,
        local_depth_ppm: None,
        local_mes: None,
        local_to_global_depth: None,
        n_transits: None,
        n_in_transit_points: None,
        n_quarters: None,
        odd_even_sigma: None,
        secondary_snr: None,
        single_event_fraction: None,
        red_noise_ratio: None,
        v_shape_score: None,
        robust_sigma_ppm: None,
        baseline_days: baseline,
        period_resolution_days: None,
        proposal_event_score: None,
        harmonic_of_rank1: None,
        brightening_max_mes: None,
        phase_scramble_max_mes: None,
        phase_scramble_evaluated: false,
        dip_brightening_mes_ratio: None,
        dip_control_mes_excess: None,
        controls_evaluated: false,
        duration_consistency_status: "NOT_RUN".to_string(),
        duration_measurable_events: 0,
        duration_consistent_fraction: None,
        duration_log2_mad: None,
        quarter_loo_status: "NOT_RUN".to_string(),
        quarter_loo_testable_quarters: 0,
        quarter_loo_positive_fraction: None,
        quarter_loo_median_depth_ratio: None,
    }
}

fn scan_light_curve(lc: &LightCurve, cfg: &Config) -> Vec<SignalRow> {
    let baseline = lc.time.last().unwrap() - lc.time.first().unwrap();
    let (sigma, mut candidates, events) = scan_candidates(lc, cfg, false);
    let preliminary_controls = candidates
        .iter()
        .any(|item| needs_controls_for_classification(&item.1, &cfg.gates));
    let preliminary_brightening = if preliminary_controls {
        brightening_control(lc, cfg, false)
    } else {
        0.0
    };
    let residual_search = cfg.max_residual_signals_per_target > 0
        && candidates
            .iter()
            .any(|item| classify(&item.1, &cfg.gates, preliminary_brightening, 0.0) == "TRANSIT_LIKE");
    if residual_search {
        let search_curve = whiten_for_search(lc, &cfg.search_whitening);
        extend_residual_candidates(lc, &search_curve, cfg, sigma, &mut candidates);
    }
    let controls_evaluated = candidates
        .iter()
        .any(|item| needs_controls_for_classification(&item.1, &cfg.gates));
    let brightening_max_mes = if residual_search && controls_evaluated {
        brightening_control(lc, cfg, true)
    } else {
        preliminary_brightening
    };
    let needs_phase_scramble = cfg.control_trials > 0
        && candidates
            .iter()
            .any(|item| classify(&item.1, &cfg.gates, brightening_max_mes, 0.0) == "TRANSIT_LIKE");
    let phase_scramble_max_mes = if needs_phase_scramble {
        phase_scramble_control(lc, cfg, residual_search)
    } else {
        0.0
    };
    if candidates.is_empty() && cfg.sparse_event_snr_min.is_none() {
        return vec![no_signal_row(lc, cfg, baseline)];
    }
    let rank1_period = candidates.first().map(|item| item.1.period);
    let mut rows: Vec<SignalRow> = candidates
        .iter()
        .enumerate()
        .map(|(index, (candidate, metrics))| {
            let legacy_classification = classify(
                metrics,
                &cfg.gates,
                brightening_max_mes,
                phase_scramble_max_mes,
            );
            let duration_validation = if legacy_classification == "TRANSIT_LIKE" {
                transit_duration_consistency(lc, candidate, metrics.robust_sigma_ppm / 1e6, cfg)
            } else {
                DurationConsistency::not_run()
            };
            let loo_validation = if legacy_classification == "TRANSIT_LIKE" {
                quarter_leave_one_out(lc, candidate, cfg)
            } else {
                QuarterLoo::not_run()
            };
            let classification = if legacy_classification == "TRANSIT_LIKE"
                && (duration_validation.status == "FAIL" || loo_validation.status == "FAIL")
            {
                "SYSTEMATIC_LIKE"
            } else {
                legacy_classification
            };
            SignalRow {
                protocol_version: cfg.protocol_version.clone(),
                source_id: lc.source_id,
                photometry: lc.photometry,
                signal_rank: index + 1,
                classification: classification.to_string(),
                search_pass: Some(candidate.search_pass),
                period_days: Some(metrics.period),
                t0_bkjd: Some(metrics.t0),
                duration_hours: Some(metrics.duration * 24.0),
                depth_ppm: Some(metrics.depth_ppm),
                search_mes: Some(candidate.search_mes),
                mes: Some(metrics.mes),
                local_depth_ppm: Some(metrics.local_depth_ppm),
                local_mes: Some(metrics.local_mes),
                local_to_global_depth: Some(metrics.local_to_global_depth),
                n_transits: Some(metrics.n_transits),
                n_in_transit_points: Some(metrics.n_in_transit_points),
                n_quarters: Some(metrics.n_quarters),
                odd_even_sigma: Some(metrics.odd_even_sigma),
                secondary_snr: Some(metrics.secondary_snr),
                single_event_fraction: Some(metrics.single_event_fraction),
                red_noise_ratio: Some(metrics.red_noise_ratio),
                v_shape_score: Some(metrics.v_shape_score),
                robust_sigma_ppm: Some(metrics.robust_sigma_ppm),
                baseline_days: baseline,
                period_resolution_days: Some(metrics.period_resolution_days),
                proposal_event_score: Some(candidate.event_score),
                harmonic_of_rank1: Some(
                    index > 0
                        && rank1_period
                            .is_some_and(|period| harmonic_related(metrics.period, period)),
                ),
                brightening_max_mes: controls_evaluated.then_some(brightening_max_mes),
                phase_scramble_max_mes: needs_phase_scramble.then_some(phase_scramble_max_mes),
                phase_scramble_evaluated: needs_phase_scramble,
                dip_brightening_mes_ratio: controls_evaluated
                    .then_some(metrics.mes / brightening_max_mes.max(1e-9)),
                dip_control_mes_excess: controls_evaluated
                    .then_some(metrics.mes - brightening_max_mes.max(phase_scramble_max_mes)),
                controls_evaluated,
                duration_consistency_status: duration_validation.status.to_string(),
                duration_measurable_events: duration_validation.measurable_events,
                duration_consistent_fraction: duration_validation.consistent_fraction,
                duration_log2_mad: duration_validation.log2_duration_mad,
                quarter_loo_status: loo_validation.status.to_string(),
                quarter_loo_testable_quarters: loo_validation.testable_quarters,
                quarter_loo_positive_fraction: loo_validation.positive_fraction,
                quarter_loo_median_depth_ratio: loo_validation.median_depth_ratio,
            }
        })
        .collect();

    if let Some(threshold) = cfg.sparse_event_snr_min {
        let mut ranked_events = events;
        ranked_events.sort_unstable_by(|left, right| {
            right.snr.partial_cmp(&left.snr).unwrap_or(Ordering::Equal)
        });
        let mut sparse_candidates = Vec::new();
        for event in ranked_events {
            if event.snr < threshold || sparse_candidates.len() == cfg.max_sparse_events_per_target
            {
                break;
            }
            let Some(metrics) = measure_single_event(lc, &event, sigma) else {
                continue;
            };
            if metrics.snr < threshold
                || metrics.n_in_transit_points < cfg.sparse_event_min_points
                || metrics.depth_ppm < cfg.gates.depth_ppm_min
                || metrics.depth_ppm > cfg.gates.depth_ppm_max
            {
                continue;
            }
            sparse_candidates.push((event, metrics));
        }
        if !sparse_candidates.is_empty() {
            let event_brightening_max = sparse_event_brightening_control(lc, cfg);
            for (event, metrics) in sparse_candidates {
                let ratio = metrics.snr / event_brightening_max.max(1e-9);
                let excess = metrics.snr - event_brightening_max;
                if !cfg
                    .gates
                    .dip_brightening_mes_ratio_min
                    .is_none_or(|minimum| ratio >= minimum)
                    || !cfg
                        .gates
                        .dip_control_mes_excess_min
                        .is_none_or(|minimum| excess >= minimum)
                {
                    continue;
                }
                let rank = rows.len() + 1;
                rows.push(SignalRow {
                    protocol_version: cfg.protocol_version.clone(),
                    source_id: lc.source_id,
                    photometry: lc.photometry,
                    signal_rank: rank,
                    classification: "EVENT_LIKE".to_string(),
                    search_pass: None,
                    period_days: None,
                    t0_bkjd: Some(event.time),
                    duration_hours: Some(event.duration * 24.0),
                    depth_ppm: Some(metrics.depth_ppm),
                    search_mes: None,
                    mes: Some(metrics.snr),
                    local_depth_ppm: Some(metrics.depth_ppm),
                    local_mes: Some(metrics.snr),
                    local_to_global_depth: Some(1.0),
                    n_transits: Some(1),
                    n_in_transit_points: Some(metrics.n_in_transit_points),
                    n_quarters: Some(1),
                    odd_even_sigma: None,
                    secondary_snr: None,
                    single_event_fraction: Some(1.0),
                    red_noise_ratio: None,
                    v_shape_score: None,
                    robust_sigma_ppm: Some(sigma * 1e6),
                    baseline_days: baseline,
                    period_resolution_days: None,
                    proposal_event_score: Some(event.snr),
                    harmonic_of_rank1: None,
                    brightening_max_mes: Some(event_brightening_max),
                    phase_scramble_max_mes: None,
                    phase_scramble_evaluated: false,
                    dip_brightening_mes_ratio: Some(ratio),
                    dip_control_mes_excess: Some(excess),
                    controls_evaluated: true,
                    duration_consistency_status: "NOT_RUN".to_string(),
                    duration_measurable_events: 1,
                    duration_consistent_fraction: None,
                    duration_log2_mad: None,
                    quarter_loo_status: "NOT_RUN".to_string(),
                    quarter_loo_testable_quarters: 0,
                    quarter_loo_positive_fraction: None,
                    quarter_loo_median_depth_ratio: None,
                });
            }
        }
    }
    if rows.is_empty() {
        rows.push(no_signal_row(lc, cfg, baseline));
    }
    rows
}

#[derive(Debug, Clone, Copy)]
struct SyntheticCase {
    label: &'static str,
    period_days: f64,
    duration_hours: f64,
    depth: f64,
}

fn synthetic_curve(case: SyntheticCase, with_transit: bool, source_id: u64) -> LightCurve {
    let cadence_days = 29.4244 / 60.0 / 24.0;
    let n = (420.0 / cadence_days) as usize;
    let t0 = 1.234;
    let duration = case.duration_hours / 24.0;
    let mut state = 0x9e37_79b9_u64;
    let mut time = Vec::with_capacity(n);
    let mut flux = Vec::with_capacity(n);
    let mut error = Vec::with_capacity(n);
    let mut quarter = Vec::with_capacity(n);
    let mut cadence = Vec::with_capacity(n);
    for index in 0..n {
        let t = index as f64 * cadence_days;
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let uniform = ((state >> 11) as f64 / ((1_u64 << 53) as f64) - 0.5) * 2.0;
        let mut value = uniform * 0.0007 + (t * 0.17).sin() * 0.00008;
        if with_transit && phase_delta(t, t0, case.period_days).abs() <= duration * 0.5 {
            value -= case.depth;
        }
        time.push(t);
        flux.push(value);
        error.push(0.0007);
        quarter.push((t / 90.0).floor() as i16);
        cadence.push(index as i32);
    }
    LightCurve {
        source_id,
        photometry: Photometry::PsfFlat,
        time,
        flux,
        error,
        quarter,
        cadence,
    }
}

fn run_self_test(cfg: &Config) -> Result<()> {
    let started = Instant::now();
    let cases = [
        SyntheticCase {
            label: "ultra_short",
            period_days: 0.31,
            duration_hours: 0.5,
            depth: 0.0040,
        },
        SyntheticCase {
            label: "short",
            period_days: std::f64::consts::PI,
            duration_hours: 1.0,
            depth: 0.0040,
        },
        SyntheticCase {
            label: "hot_jupiter",
            period_days: 7.3123,
            duration_hours: 3.0,
            depth: 0.0035,
        },
        SyntheticCase {
            label: "warm_neptune",
            period_days: 29.75,
            duration_hours: 6.0,
            depth: 0.0025,
        },
        SyntheticCase {
            label: "long_period",
            period_days: 73.2,
            duration_hours: 10.0,
            depth: 0.0035,
        },
        SyntheticCase {
            label: "long_duration",
            period_days: 140.0,
            duration_hours: 18.0,
            depth: 0.0040,
        },
    ];
    let supported: Vec<SyntheticCase> = cases
        .into_iter()
        .filter(|case| {
            case.period_days >= cfg.period_min_days
                && case.period_days <= cfg.period_max_days
                && cfg
                    .duration_hours
                    .iter()
                    .any(|duration| (duration - case.duration_hours).abs() < 1e-9)
        })
        .collect();
    if supported.is_empty() {
        bail!("configuration supports none of the deterministic injection cases")
    }

    let mut summaries = Vec::with_capacity(supported.len());
    for (index, case) in supported.iter().copied().enumerate() {
        let injected = synthetic_curve(case, true, 10 + index as u64);
        let (_, recovered, _) = scan_candidates(&injected, cfg, true);
        let best = recovered
            .iter()
            .min_by(|left, right| {
                (left.1.period - case.period_days)
                    .abs()
                    .partial_cmp(&(right.1.period - case.period_days).abs())
                    .unwrap_or(Ordering::Equal)
            })
            .with_context(|| format!("{} injection produced no candidate", case.label))?;
        let relative_error = (best.1.period / case.period_days - 1.0).abs();
        if relative_error > 0.01 || best.1.mes < cfg.gates.mes_min {
            let recovered_summary: Vec<(f64, f64)> = recovered
                .iter()
                .map(|item| (item.1.period, item.1.mes))
                .collect();
            bail!(
                "{} injection failed: period={} MES={} relative_error={} candidates={:?}",
                case.label,
                best.1.period,
                best.1.mes,
                relative_error,
                recovered_summary
            )
        }
        let injected_brightening = brightening_control(&injected, cfg, true);
        let injected_scramble = phase_scramble_control(&injected, cfg, true);
        if classify(&best.1, &cfg.gates, injected_brightening, injected_scramble) != "TRANSIT_LIKE" {
            bail!(
                "{} injection recovered but failed enhanced classification: MES={} local_MES={} brightening_MES={} scramble_MES={}",
                case.label,
                best.1.mes,
                best.1.local_mes,
                injected_brightening,
                injected_scramble
            )
        }
        let duration_validation =
            transit_duration_consistency(&injected, &best.0, best.1.robust_sigma_ppm / 1e6, cfg);
        if cfg.survivor_validation.duration_consistency.enabled
            && duration_validation.status != "PASS"
        {
            bail!(
                "{} injection failed duration consistency: status={} measurable={} fraction={:?}",
                case.label,
                duration_validation.status,
                duration_validation.measurable_events,
                duration_validation.consistent_fraction
            )
        }
        let loo_validation = quarter_leave_one_out(&injected, &best.0, cfg);
        if cfg.survivor_validation.quarter_loo.enabled && loo_validation.status != "PASS" {
            bail!(
                "{} injection failed quarter LOO: status={} testable={} positive_fraction={:?} median_depth_ratio={:?}",
                case.label,
                loo_validation.status,
                loo_validation.testable_quarters,
                loo_validation.positive_fraction,
                loo_validation.median_depth_ratio
            )
        }
        summaries.push(format!(
            "{{\"label\":\"{}\",\"injection_period_days\":{:.8},\"recovered_period_days\":{:.8},\"duration_hours\":{:.3},\"relative_error\":{:.8},\"mes\":{:.3}}}",
            case.label,
            case.period_days,
            best.1.period,
            case.duration_hours,
            relative_error,
            best.1.mes
        ));
    }

    let null_case = supported[0];
    let null = synthetic_curve(null_case, false, 2);
    let (_, null_candidates, _) = scan_candidates(&null, cfg, true);
    let null_transit_like = null_candidates
        .iter()
        .any(|(_, metrics)| classify(metrics, &cfg.gates, 0.0, 0.0) == "TRANSIT_LIKE");
    if null_transit_like {
        bail!("deterministic null produced a TRANSIT_LIKE classification")
    }
    println!(
        "{{\"injections\":[{}],\"null_transit_like\":false,\"elapsed_seconds\":{:.3}}}",
        summaries.join(","),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn encoded_curve(view: u64, records: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&123_u64.to_le_bytes());
        bytes.extend_from_slice(&view.to_le_bytes());
        bytes.extend_from_slice(&records.to_le_bytes());
        for index in 0..records {
            bytes.extend_from_slice(&(index as f64 * 0.02).to_le_bytes());
            bytes.extend_from_slice(&0.0_f32.to_le_bytes());
            bytes.extend_from_slice(&0.001_f32.to_le_bytes());
            bytes.extend_from_slice(&1_i16.to_le_bytes());
            bytes.extend_from_slice(&(index as i32).to_le_bytes());
        }
        bytes
    }

    fn test_directory(label: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .unwrap();
        let path = root
            .join("dump/rust-multiview-tests-20260912")
            .join(format!("{label}-{}-{stamp}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_test_curve(folder: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let path = folder.join(name);
        fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn fsc_lc2_preserves_all_four_view_identities() {
        let folder = test_directory("four-views");
        for (code, label) in ["psf_flat", "psf_qdiag", "sap_qdiag", "psf_nova"]
            .iter()
            .enumerate()
        {
            let path = write_test_curve(
                &folder,
                &format!("123.{label}.fsc"),
                &encoded_curve(code as u64, 100),
            );
            let curve = read_light_curve(&path).unwrap();
            assert_eq!(curve.source_id, 123);
            assert_eq!(curve.photometry as usize, code);
            assert_eq!(serde_json::to_value(curve.photometry).unwrap(), *label);
            assert_eq!(curve.time.len(), 100);
            assert_eq!(curve.cadence[99], 99);
        }
        assert_eq!(scan_inputs(&folder).unwrap().len(), 4);
    }

    #[test]
    fn fsc_lc2_rejects_unknown_view_and_malformed_headers() {
        let good = encoded_curve(0, 100);
        let mut cases = Vec::new();
        let mut legacy = good.clone();
        legacy[..8].copy_from_slice(b"FSC_LC1\0");
        cases.push(legacy);
        cases.push(encoded_curve(4, 100));
        cases.push(encoded_curve(0, 99));
        cases.push(good[..20].to_vec());
        cases.push(good[..good.len() - 1].to_vec());
        let mut trailing = good.clone();
        trailing.push(0);
        cases.push(trailing);
        let mut oversized = good;
        oversized[24..32].copy_from_slice(&u64::MAX.to_le_bytes());
        cases.push(oversized);
        for bytes in cases {
            assert!(read_curve_header(&mut Cursor::new(&bytes), bytes.len() as u64).is_err());
        }
    }

    #[test]
    fn fsc_lc2_rejects_invalid_photometry_records() {
        let folder = test_directory("invalid-records");
        let good = encoded_curve(0, 100);
        let mut nonfinite = good.clone();
        nonfinite[32..40].copy_from_slice(&f64::NAN.to_le_bytes());
        let mut zero_error = good.clone();
        zero_error[44..48].copy_from_slice(&0.0_f32.to_le_bytes());
        let mut unsorted = good;
        unsorted[54..62].copy_from_slice(&(-1.0_f64).to_le_bytes());
        for (index, bytes) in [nonfinite, zero_error, unsorted].iter().enumerate() {
            let path = write_test_curve(&folder, &format!("invalid-{index}.fsc"), bytes);
            assert!(read_light_curve(&path).is_err());
        }
    }

    #[test]
    fn duplicate_source_view_is_rejected_before_scanning() {
        let folder = test_directory("duplicate-view");
        write_test_curve(&folder, "first.fsc", &encoded_curve(2, 100));
        write_test_curve(&folder, "different-name.fsc", &encoded_curve(2, 100));
        let error = scan_inputs(&folder).unwrap_err().to_string();
        assert!(error.contains("duplicate source/view"), "{error}");
    }

    #[test]
    fn multiview_no_signal_output_is_sorted_and_complete() {
        let folder = test_directory("scan-four-views");
        for code in (0..4).rev() {
            write_test_curve(
                &folder,
                &format!("123.{code}.fsc"),
                &encoded_curve(code, 100),
            );
        }
        let output = folder.join("result.csv");
        run_scan(ScanArgs {
            config: Path::new(env!("CARGO_MANIFEST_DIR")).join("config.json"),
            input_dir: folder,
            output: output.clone(),
            photometry: None,
        })
        .unwrap();
        let mut reader = csv::Reader::from_path(output).unwrap();
        let headers = reader.headers().unwrap().clone();
        let view_index = headers
            .iter()
            .position(|name| name == "photometry")
            .unwrap();
        let class_index = headers
            .iter()
            .position(|name| name == "classification")
            .unwrap();
        let rows: Vec<_> = reader.records().map(|row| row.unwrap()).collect();
        assert_eq!(rows.len(), 4);
        let labels: Vec<_> = rows.iter().map(|row| &row[view_index]).collect();
        assert_eq!(labels, ["psf_flat", "psf_qdiag", "sap_qdiag", "psf_nova"]);
        assert!(rows.iter().all(|row| &row[class_index] == "NO_SIGNAL"));
    }

    #[test]
    fn scan_can_select_one_declared_photometry() {
        let folder = test_directory("scan-primary-view");
        for code in 0..4 {
            write_test_curve(
                &folder,
                &format!("123.{code}.fsc"),
                &encoded_curve(code, 100),
            );
        }
        let output = folder.join("result.csv");
        run_scan(ScanArgs {
            config: Path::new(env!("CARGO_MANIFEST_DIR")).join("config.json"),
            input_dir: folder,
            output: output.clone(),
            photometry: Some("psf_flat".to_string()),
        })
        .unwrap();
        let mut reader = csv::Reader::from_path(output).unwrap();
        let view_index = reader
            .headers()
            .unwrap()
            .iter()
            .position(|name| name == "photometry")
            .unwrap();
        let rows: Vec<_> = reader.records().map(|row| row.unwrap()).collect();
        assert_eq!(rows.len(), 1);
        assert_eq!(&rows[0][view_index], "psf_flat");
    }

    #[test]
    fn output_reconciliation_rejects_missing_or_duplicate_view_ranks() {
        let folder = test_directory("reconcile");
        let path = write_test_curve(&folder, "123.sap.fsc", &encoded_curve(2, 100));
        let inputs = scan_inputs(&folder).unwrap();
        let curve = read_light_curve(&path).unwrap();
        let config = balanced_config();
        let mut rows = scan_light_curve(&curve, &config);
        assert!(reconcile_rows(&inputs, &rows).is_ok());
        assert!(reconcile_rows(&inputs, &[]).is_err());
        rows.extend(scan_light_curve(&curve, &config));
        assert!(reconcile_rows(&inputs, &rows).is_err());
    }

    #[test]
    fn photometry_labels_leave_numerical_detection_unchanged() {
        let mut config = balanced_config();
        config.max_residual_signals_per_target = 0;
        let case = SyntheticCase {
            label: "view-equivalence",
            period_days: 7.3123,
            duration_hours: 3.0,
            depth: 0.01,
        };
        let mut curve = synthetic_curve(case, true, 903);
        let mut reference = None;
        for code in 0..4 {
            curve.photometry = Photometry::from_code(code).unwrap();
            assert_eq!(
                subset_by_quarter(&curve, 1, true).photometry,
                curve.photometry
            );
            assert_eq!(
                rotated_quarter_control(&curve, 0).photometry,
                curve.photometry
            );
            let rows = scan_light_curve(&curve, &config);
            assert!(rows.iter().all(|row| row.photometry == curve.photometry));
            let mut values = serde_json::to_value(rows).unwrap();
            for row in values.as_array_mut().unwrap() {
                row.as_object_mut().unwrap().remove("photometry");
            }
            if let Some(expected) = &reference {
                assert_eq!(&values, expected);
            } else {
                reference = Some(values);
            }
        }
    }

    #[test]
    fn phase_delta_wraps() {
        assert!((phase_delta(9.9, 0.0, 10.0) + 0.1).abs() < 1e-9);
        assert!((phase_delta(10.1, 0.0, 10.0) - 0.1).abs() < 1e-9);
    }

    #[test]
    fn period_precision_uses_timing_leverage() {
        let left = Event {
            time: 0.0,
            duration: 0.2,
            snr: 5.0,
        };
        let right = Event {
            time: 100.0,
            ..left
        };
        let short_baseline = pair_period_precision_weight(&left, &right, 1);
        let long_baseline = pair_period_precision_weight(&left, &right, 10);
        assert!((long_baseline / short_baseline - 100.0).abs() < 1e-9);
    }

    #[test]
    fn extension_candidates_do_not_consume_core_budget() {
        let candidate = |period, search_mes| BoxCandidate {
            period,
            t0: 1.0,
            duration: 0.1,
            search_mes,
            event_score: 1.0,
            search_pass: 1,
        };
        let candidates = [
            candidate(150.0, 100.0),
            candidate(2.0, 10.0),
            candidate(3.0, 9.0),
            candidate(0.3, 8.0),
        ];
        let core =
            select_candidates_where(&candidates, 2, |period| (0.5..=100.0).contains(&period));
        assert_eq!(core.len(), 2);
        assert_eq!(core[0].period, 2.0);
        assert_eq!(core[1].period, 3.0);
    }

    #[test]
    fn periodic_limit_requires_three_epochs() {
        let case = SyntheticCase {
            label: "period_limit",
            period_days: 7.0,
            duration_hours: 3.0,
            depth: 0.01,
        };
        let curve = synthetic_curve(case, false, 907);
        let mut cfg = balanced_config();
        cfg.period_max_days = 1000.0;
        assert!((periodic_max_days(&curve, &cfg) - 210.0).abs() < 0.1);
    }

    #[test]
    fn median_selects_odd_and_even_centers() {
        assert_eq!(median(&[9.0, 1.0, 5.0]), 5.0);
        assert_eq!(median(&[9.0, 1.0, 7.0, 3.0]), 5.0);
        assert_eq!(median(&[f64::NAN, 7.0, 3.0]), 5.0);
    }

    #[test]
    fn robust_scale_ignores_one_outlier() {
        let sigma = robust_sigma(&[-1.0, -0.9, 0.0, 0.9, 1.0, 100.0]);
        assert!(sigma > 1.0 && sigma < 2.0);
    }

    fn balanced_config() -> Config {
        serde_json::from_str(include_str!("../config.json")).unwrap()
    }

    fn inject_events(curve: &mut LightCurve, times: &[f64], duration_hours: f64, depth: f64) {
        let half_duration = duration_hours / 48.0;
        for (time, flux) in curve.time.iter().zip(&mut curve.flux) {
            if times
                .iter()
                .any(|center| (*time - *center).abs() <= half_duration)
            {
                *flux -= depth;
            }
        }
    }

    fn sparse_config() -> Config {
        let mut cfg = balanced_config();
        cfg.control_trials = 0;
        cfg.sparse_event_snr_min = Some(8.0);
        cfg.sparse_event_min_points = 3;
        cfg.max_sparse_events_per_target = 3;
        cfg
    }

    #[test]
    fn sparse_lane_retains_one_transit_without_a_period() {
        let case = SyntheticCase {
            label: "single_transit",
            period_days: 50.0,
            duration_hours: 6.0,
            depth: 0.0,
        };
        let mut curve = synthetic_curve(case, false, 904);
        inject_events(&mut curve, &[210.0], case.duration_hours, 0.012);
        let rows = scan_light_curve(&curve, &sparse_config());
        let event = rows
            .iter()
            .find(|row| row.classification == "EVENT_LIKE")
            .unwrap();
        assert!(event.period_days.is_none());
        assert!((event.t0_bkjd.unwrap() - 210.0).abs() <= case.duration_hours / 24.0);
    }

    #[test]
    fn sparse_lane_retains_strong_ttv_events() {
        let case = SyntheticCase {
            label: "strong_ttv",
            period_days: 20.0,
            duration_hours: 6.0,
            depth: 0.0,
        };
        let injected = [50.0, 71.5, 89.0, 112.5];
        let mut curve = synthetic_curve(case, false, 905);
        inject_events(&mut curve, &injected, case.duration_hours, 0.012);
        let rows = scan_light_curve(&curve, &sparse_config());
        let retained: Vec<f64> = rows
            .iter()
            .filter(|row| row.classification == "EVENT_LIKE")
            .filter_map(|row| row.t0_bkjd)
            .collect();
        assert!(retained.len() >= 2, "retained={retained:?}");
        assert!(retained.iter().all(|time| {
            injected
                .iter()
                .any(|expected| (*time - *expected).abs() <= case.duration_hours / 24.0)
        }));
    }

    #[test]
    fn duration_consistency_rejects_alternating_widths() {
        let cfg = balanced_config();
        let case = SyntheticCase {
            label: "variable_duration",
            period_days: 7.3123,
            duration_hours: 3.0,
            depth: 0.01,
        };
        let mut curve = synthetic_curve(case, false, 901);
        let t0 = 1.234;
        for (time, flux) in curve.time.iter().zip(&mut curve.flux) {
            let epoch = ((*time - t0) / case.period_days).round() as i64;
            let factor = if epoch.rem_euclid(2) == 0 { 0.16 } else { 4.0 };
            if phase_delta(*time, t0, case.period_days).abs()
                <= case.duration_hours / 24.0 * factor * 0.5
            {
                *flux -= case.depth;
            }
        }
        let candidate = BoxCandidate {
            period: case.period_days,
            t0,
            duration: case.duration_hours / 24.0,
            search_mes: 20.0,
            event_score: 10.0,
            search_pass: 1,
        };
        let outcome =
            transit_duration_consistency(&curve, &candidate, robust_sigma(&curve.flux), &cfg);
        assert_eq!(outcome.status, "FAIL", "{outcome:?}");
    }

    #[test]
    fn quarter_loo_rejects_single_quarter_artifact() {
        let cfg = balanced_config();
        let case = SyntheticCase {
            label: "quarter_artifact",
            period_days: 7.3123,
            duration_hours: 3.0,
            depth: 0.01,
        };
        let mut curve = synthetic_curve(case, false, 902);
        let t0 = 1.234;
        for index in 0..curve.time.len() {
            if curve.quarter[index] == 1
                && phase_delta(curve.time[index], t0, case.period_days).abs()
                    <= case.duration_hours / 24.0 * 0.5
            {
                curve.flux[index] -= case.depth;
            }
        }
        let candidate = BoxCandidate {
            period: case.period_days,
            t0,
            duration: case.duration_hours / 24.0,
            search_mes: 20.0,
            event_score: 10.0,
            search_pass: 1,
        };
        let outcome = quarter_leave_one_out(&curve, &candidate, &cfg);
        assert_eq!(outcome.status, "FAIL", "{outcome:?}");
    }
}
