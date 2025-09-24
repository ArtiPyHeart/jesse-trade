_fracdiff_features = []
for p1 in ["o", "h", "l", "c"]:
    for p2 in ["o", "h", "l", "c"]:
        for l in range(1, 6):
            _fracdiff_features.append(f"frac_{p1}_{p2}{l}_diff")

BUILDIN_FEATURES = (
    _fracdiff_features
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
)
