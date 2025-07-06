function kurtosis_loss(tree, dataset::Dataset{T,L}, options) where {T,L}
    using NPZ
    using Statistics

    candles = npzread("btc_1m.npy")
    candles = candles[candles[:, 6].>0, :]

    # build bar function
    function build_bar_by_cumsum(candles, condition, threshold)
        n = size(candles, 1)
        bars = zeros(Float64, n, 6)
        bar_index = 1

        # 初始化第一个bar
        bar_timestamp = candles[1, 1]
        bar_open = candles[1, 2]
        bar_close = candles[1, 3]
        bar_high = candles[1, 4]
        bar_low = candles[1, 5]
        bar_volume = candles[1, 6]
        bar_cumsum = condition[1]

        for i in 2:n
            if bar_cumsum <= threshold
                bar_cumsum += condition[i]
                bar_timestamp = max(bar_timestamp, candles[i, 1])
                bar_volume += candles[i, 6]
                bar_high = max(bar_high, candles[i, 4])
                bar_low = min(bar_low, candles[i, 5])
                bar_close = candles[i, 3]
            else
                bars[bar_index, 1] = bar_timestamp
                bars[bar_index, 2] = bar_open
                bars[bar_index, 3] = bar_close
                bars[bar_index, 4] = bar_high
                bars[bar_index, 5] = bar_low
                bars[bar_index, 6] = bar_volume
                bar_index += 1

                bar_timestamp = candles[i, 1]
                bar_open = candles[i, 2]
                bar_close = candles[i, 3]
                bar_high = candles[i, 4]
                bar_low = candles[i, 5]
                bar_volume = candles[i, 6]
                bar_cumsum = condition[i]
            end
        end

        return bars[1:bar_index-1, :]
    end

    function compute_kurtosis(merged_bars)
        n = size(merged_bars, 1)
        if n <= 5
            return Inf
        end

        close_prices = merged_bars[:, 3]
        log_returns = log.(close_prices[6:end] ./ close_prices[1:end-5])

        if length(log_returns) == 0
            return Inf
        end

        mean_ret = mean(log_returns)
        std_ret = std(log_returns)
        if std_ret == 0
            return Inf
        end

        standardized = (log_returns .- mean_ret) ./ std_ret
        n = length(standardized)
        kurt = sum(standardized .^ 4) / n

        return kurt
    end

    # Require root node to be binary, so we can split it,
    # otherwise return a large loss:
    tree.degree != 2 && return L(Inf)

    P = tree.l
    Q = tree.r

    # Evaluate numerator:
    P_prediction, flag = eval_tree_array(P, dataset.X, options)
    !flag && return L(Inf)

    # Evaluate denominator:
    Q_prediction, flag = eval_tree_array(Q, dataset.X, options)
    !flag && return L(Inf)

    # Impose functional form:
    prediction = P_prediction ./ Q_prediction

    # @assert length(prediction) == length(candles) "prediction length: $(length(prediction)) != candles length: $(length(candles))"
    len_gap = length(candles) - length(prediction)
    candles = candles[1+len_gap:end, :]

    cumsum_threshold = sum(prediction) / (length(prediction) ÷ 120)
    merged_bars = build_bar_by_cumsum(candles, prediction, cumsum_threshold)

    # 如果合并后的bar数量小于原始bar数量太多，则返回Inf
    if length(merged_bars) < length(candles) ÷ 300
        return L(Inf)
    end

    # 计算总体kurtosis
    kurtosis_all = compute_kurtosis(merged_bars)
    # 计算前1/3数据的kurtosis
    kurtosis_first_third = compute_kurtosis(merged_bars[1:end÷3, :])
    # 计算中间1/3数据的kurtosis
    kurtosis_middle_third = compute_kurtosis(merged_bars[end÷3:2*end÷3, :])
    # 计算后1/3数据的kurtosis
    kurtosis_last_third = compute_kurtosis(merged_bars[end÷3:end, :])

    kurtosis = mean([
        kurtosis_all,
        kurtosis_first_third,
        kurtosis_middle_third,
        kurtosis_last_third
    ])
    return L(kurtosis)
end