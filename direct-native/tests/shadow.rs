use std::path::Path;

use direct_native::shadow::{
    render_timing_summary, ChainExecutionTimes, ParityAuthority, RasterSourceMode, ShadowReport,
};

#[test]
fn parity_artifacts_live_under_stage_directory() {
    let stage = Path::new("target/raster/chains/run/prefill_range_l3");
    assert_eq!(
        direct_native::shadow::parity_dir(stage),
        stage.join("direct-native-parity")
    );
}

#[test]
fn timing_summary_annotates_selected_stage() {
    let timings: ChainExecutionTimes = serde_json::from_str(
        r#"{
            "version": 1,
            "stages": [{
                "name": "prefill_range_l3",
                "exec_duration_ns": 2000000000
            }],
            "total_exec_duration_ns": 2000000000
        }"#,
    )
    .unwrap();
    let shadow = ShadowReport {
        version: 3,
        stage: "prefill_range_l3".into(),
        routine: "prefill_range".into(),
        instance: Some(3),
        authority: ParityAuthority::NonAuthoritative,
        raster_source_mode: RasterSourceMode::Unauthenticated,
        matched: true,
        input_load_duration_ns: 10_000_000,
        kernel_duration_ns: 200_000_000,
        encode_write_duration_ns: 20_000_000,
        direct_stage_duration_ns: 250_000_000,
    };
    let encoded = serde_json::to_vec(&shadow).unwrap();
    let json = String::from_utf8(encoded.clone()).unwrap();
    assert!(json.contains(r#""authority":"non_authoritative""#));
    assert!(json.contains(r#""raster_source_mode":"unauthenticated""#));
    let shadow: ShadowReport = serde_json::from_slice(&encoded).unwrap();

    let summary = render_timing_summary(&timings, &shadow).unwrap();
    assert!(summary.contains("prefill_range_l3"));
    assert!(summary.contains("250.00ms"));
    assert!(summary.contains("8.00x"));
    assert!(summary.contains("MATCH"));
    assert!(summary.contains("NON-AUTHORITATIVE"));
    assert!(summary.contains("hybrid pipeline: --no-auth"));
}
