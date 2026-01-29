def biased_ema(data, smoothing_weight=0.99):
    smoothed_data = []
    last = 0
    num_accum = 0
    for next_val in data:
        num_accum += 1
        last = last * smoothing_weight + (1 - smoothing_weight) * next_val
        debias_weight = 1.0 - pow(smoothing_weight, num_accum)
        smoothed_data.append(last / debias_weight)
    return smoothed_data