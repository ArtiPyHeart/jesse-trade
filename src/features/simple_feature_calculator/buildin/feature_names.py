_fracdiff_features = []
for p1 in ["o", "h", "l", "c"]:
    for p2 in ["o", "h", "l", "c"]:
        for l in range(1, 6):
            _fracdiff_features.append(f"frac_{p1}_{p2}{l}_diff")

BUILDIN_FEATURES = (
    _fracdiff_features
    # WorldQuant 101 Alphas
    + [
        "wq_alpha_001",
        "wq_alpha_002",
        "wq_alpha_003",
        "wq_alpha_004",
        "wq_alpha_005",
        "wq_alpha_006",
        "wq_alpha_007",
        "wq_alpha_008",
        "wq_alpha_009",
        "wq_alpha_010",
        "wq_alpha_011",
        "wq_alpha_012",
        "wq_alpha_013",
        "wq_alpha_014",
        "wq_alpha_015",
        "wq_alpha_016",
        "wq_alpha_017",
        "wq_alpha_018",
        "wq_alpha_019",
        "wq_alpha_020",
        "wq_alpha_021",
        "wq_alpha_022",
        "wq_alpha_023",
        "wq_alpha_024",
        "wq_alpha_025",
        "wq_alpha_026",
        "wq_alpha_027",
        "wq_alpha_030",
        "wq_alpha_033",
        "wq_alpha_034",
        "wq_alpha_035",
        "wq_alpha_036",
        "wq_alpha_037",
        "wq_alpha_038",
        "wq_alpha_039",
        "wq_alpha_040",
        "wq_alpha_041",
        "wq_alpha_042",
        "wq_alpha_043",
        "wq_alpha_044",
        "wq_alpha_045",
        "wq_alpha_046",
        "wq_alpha_047",
        "wq_alpha_049",
        "wq_alpha_050",
        "wq_alpha_051",
        "wq_alpha_052",
        "wq_alpha_053",
        "wq_alpha_054",
        "wq_alpha_055",
        "wq_alpha_057",
        "wq_alpha_061",
        "wq_alpha_062",
        "wq_alpha_064",
        "wq_alpha_065",
        "wq_alpha_066",
        "wq_alpha_068",
        "wq_alpha_071",
        "wq_alpha_072",
        "wq_alpha_073",
        "wq_alpha_074",
        "wq_alpha_075",
        "wq_alpha_077",
        "wq_alpha_078",
        "wq_alpha_081",
        "wq_alpha_083",
        "wq_alpha_084",
        "wq_alpha_085",
        "wq_alpha_086",
        "wq_alpha_088",
        "wq_alpha_092",
        "wq_alpha_094",
        "wq_alpha_095",
        "wq_alpha_096",
        "wq_alpha_098",
        "wq_alpha_099",
        "wq_alpha_101",
    ]
    # 自定义轴持续时间
    + ["bar_duration"]
    + ["adx_7", "adx_14", "aroon_diff"]
    + [f"ac_{i}" for i in range(47)]
    + ["acc_swing_index"]
    + [f"acp_pwr_{i}" for i in range(39)]
    + [
        "acr",
        "adaptive_bp_0",
        "adaptive_bp_1",
        "adaptive_cci",
        "adaptive_rsi",
        "adaptive_stochastic",
        "amihud_lambda",
        "bandpass_0",
        "bandpass_1",
        "bekker_parkinson_vol",
        "chaiken_money_flow",
        "change_variance_ratio",
        "cmma",
        "corwin_schultz_estimator",
    ]
    + [f"comb_spectrum_{i}" for i in range(40)]
    + [f"conv_{i}" for i in range(46)]
    + [f"dft_{i}" for i in range(40)]
    + ["dual_diff", "ehlers_early_onset_trend"]
    + [f"sample_entropy_w{w}_spot" for w in [32, 64, 128, 256]]
    + [f"sample_entropy_w{w}_array" for w in [32, 64, 128, 256]]
    + [f"approximate_entropy_w{w}_spot" for w in [32, 64, 128, 256]]
    + [f"approximate_entropy_w{w}_array" for w in [32, 64, 128, 256]]
    + [
        "entropy_for_jesse",
        "evenbetter_sinewave_long",
        "evenbetter_sinewave_short",
        "fisher",
        "fti_0",
        "fti_1",
        "forecast_oscillator",
        "hasbrouck_lambda",
        "homodyne",
        "hurst_coef_fast",
        "hurst_coef_slow",
        "iqr_ratio",
        "kyle_lambda",
        "ma_difference",
        "mod_rsi",
        "mod_stochastic",
        "natr",
        "norm_on_balance_volume",
        "phase_accumulation",
        "pfe",
        "price_change_oscillator",
        "price_variance_ratio",
        "reactivity",
        "roll_impact",
        "roll_measure",
        "roofing_filter",
        "stc",
    ]
    + [f"swamicharts_rsi_{i}" for i in range(44)]
    + [f"swamicharts_stochastic_{i}" for i in range(44)]
    + ["trendflex", "voss_0", "voss_1", "vwap", "williams_r"]
    + [f"cwt_w{w}_{i}" for w in [32, 64, 128, 256] for i in range(21)]
    + [f"vmd_w{w}_{i}" for w in [32, 64, 128, 256] for i in range(3)]
    # Market Behavior Features
    + [
        "reverse_1v5",
        "reverse_2v10",
        "reverse_3v15",
        "re_3",
        "re_5",
        "reverse_ma5",
        "reverse_ma10",
        "overbuy_high5",
        "oversell_low5",
        "hl_diff",
        "hl_diff_ma5",
        "over_volatility",
    ]
)

if __name__ == "__main__":
    assert len(BUILDIN_FEATURES) == len(
        list(set(BUILDIN_FEATURES))
    ), "duplicate buildin features"
