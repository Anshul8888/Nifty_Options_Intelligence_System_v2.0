import numpy as np
from scipy.stats import norm
from scipy.fft import fft
import config


# ================================================================
# SECTION 1: BLACK-SCHOLES & GREEKS
# ================================================================

def bs_price(S, K, T, r, sigma, option_type="CE"):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if option_type == "CE" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "CE":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_vega(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def implied_volatility(market_price, S, K, T, r, option_type="CE"):
    if market_price <= 0 or T <= 0:
        return None
    intrinsic = max(S - K, 0) if option_type == "CE" else max(K - S, 0)
    if market_price < intrinsic - 0.5:
        return None
    sigma = 0.3
    for _ in range(100):
        price = bs_price(S, K, T, r, sigma, option_type)
        vega = bs_vega(S, K, T, r, sigma)
        if vega < 1e-10:
            break
        diff = market_price - price
        if abs(diff) < 0.01:
            break
        sigma += diff / vega
        sigma = max(0.01, min(sigma, 5.0))
    return sigma if 0.01 < sigma < 5.0 else None


def calculate_delta(S, K, T, r, sigma, option_type="CE"):
    if T <= 0 or sigma is None or sigma <= 0:
        return (1.0 if S > K else 0.0) if option_type == "CE" else (-1.0 if S < K else 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == "CE" else norm.cdf(d1) - 1.0


def calculate_theta(S, K, T, r, sigma, option_type="CE"):
    if T <= 0 or sigma is None or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    common = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "CE":
        theta = common - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta = common + r * K * np.exp(-r * T) * norm.cdf(-d2)
    return theta / 365


# ================================================================
# SECTION 2: KALMAN FILTER
# ================================================================

class KalmanFilter1D:
    def __init__(self, initial_value, Q=1.0, R=2.0):
        self.x = initial_value
        self.P = 10.0
        self.Q = Q
        self.R = R

    def update(self, measurement):
        x_pred = self.x
        P_pred = self.P + self.Q
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
        return self.x


# ================================================================
# SECTION 3: ANALYSIS STATE
# ================================================================

class AnalysisState:
    def __init__(self):
        self.kalman_filters = {}
        self.regime_scores = {
            "trending_calm": 0.25,
            "trending_volatile": 0.25,
            "ranging_calm": 0.25,
            "ranging_volatile": 0.25
        }
        self.current_regime = "unknown"
        self.option_price_history = []
        self.nifty_return_bins = []
        self.option_return_bins = {}
        self.prev_implied_dist = None
        self.prev_dist_mean = None
        self.prev_dist_std = None
        self.active_signals = []
        self.candle_count = 0
        self.last_hurst = 0.5
        self.pca_explained = None
        self.pca_direction = 0
        self.nifty_spot = 0

    def get_kalman(self, key, initial_value):
        if key not in self.kalman_filters:
            spread = abs(initial_value * 0.01) + 0.5
            self.kalman_filters[key] = KalmanFilter1D(
                initial_value,
                Q=config.KALMAN_PROCESS_NOISE,
                R=max(spread, config.KALMAN_MEASUREMENT_NOISE)
            )
        return self.kalman_filters[key]

    def store_option_prices(self, enriched_chain, nifty_price):
        self.nifty_spot = nifty_price
        snapshot = {"nifty": nifty_price}
        for strike, types in enriched_chain.items():
            for opt_type in ["CE", "PE"]:
                if opt_type in types:
                    key = f"{strike}_{opt_type}"
                    snapshot[key] = types[opt_type].get("ltp", 0)
        self.option_price_history.append(snapshot)
        if len(self.option_price_history) > 50:
            self.option_price_history = self.option_price_history[-50:]

    def store_binned_returns(self, nifty_closes, enriched_chain, atm_strike):
        if len(nifty_closes) < 2:
            return
        nifty_ret = (nifty_closes[-1] - nifty_closes[-2]) / nifty_closes[-2]
        thr = config.TE_RETURN_THRESHOLD
        nifty_bin = 2 if nifty_ret > thr else (0 if nifty_ret < -thr else 1)
        self.nifty_return_bins.append(nifty_bin)
        if len(self.nifty_return_bins) > 50:
            self.nifty_return_bins = self.nifty_return_bins[-50:]

        near_strikes = [s for s in enriched_chain.keys() if abs(s - atm_strike) <= 300]
        for strike in near_strikes:
            for opt_type in ["CE", "PE"]:
                key = f"{strike}_{opt_type}"
                if len(self.option_price_history) >= 2:
                    prev_price = self.option_price_history[-2].get(key, 0)
                    curr_price = self.option_price_history[-1].get(key, 0)
                    if prev_price > 0 and curr_price > 0:
                        opt_ret = (curr_price - prev_price) / prev_price
                        opt_bin = 2 if opt_ret > thr else (0 if opt_ret < -thr else 1)
                        if key not in self.option_return_bins:
                            self.option_return_bins[key] = []
                        self.option_return_bins[key].append(opt_bin)
                        if len(self.option_return_bins[key]) > 50:
                            self.option_return_bins[key] = self.option_return_bins[key][-50:]

    def increment(self):
        self.candle_count += 1


# ================================================================
# SECTION 4: CHAIN ENRICHMENT (with Kalman)
# ================================================================

def enrich_chain(snapshot, days_to_expiry, state=None):
    S = snapshot["nifty_price"]
    T = days_to_expiry
    r = config.RISK_FREE_RATE
    enriched = {}

    for strike, types in snapshot["options"].items():
        enriched[strike] = {}
        for opt_type in ["CE", "PE"]:
            if opt_type not in types:
                continue
            data = types[opt_type].copy()
            ltp = data["ltp"]

            if state and ltp > 0:
                kf_key = (strike, opt_type)
                kf = state.get_kalman(kf_key, ltp)
                spread = data.get("spread", 1.0)
                kf.R = max(spread * 0.5, 0.5)
                data["kalman_price"] = kf.update(ltp)
            else:
                data["kalman_price"] = ltp

            iv = implied_volatility(ltp, S, strike, T, r, opt_type)
            data["iv"] = iv
            sigma = iv if iv else 0.2
            data["delta"] = calculate_delta(S, strike, T, r, sigma, opt_type)
            data["theta"] = calculate_theta(S, strike, T, r, sigma, opt_type)

            bid = data.get("bid", 0)
            ask = data.get("ask", 0)
            data["spread"] = ask - bid if ask > 0 and bid > 0 else 999

            enriched[strike][opt_type] = data

    return enriched


# ================================================================
# SECTION 5: LINEAR ALGEBRA — PCA
# ================================================================

def pca_chain_analysis(state, enriched_chain, atm_strike):
    if len(state.option_price_history) < config.PCA_MIN_SNAPSHOTS:
        return {"explained_ratio": None, "direction": 0, "anomaly": False}

    near_strikes = sorted([s for s in enriched_chain.keys() if abs(s - atm_strike) <= 300])
    columns = []
    for strike in near_strikes:
        for opt_type in ["CE", "PE"]:
            columns.append(f"{strike}_{opt_type}")

    n_time = min(len(state.option_price_history), 20)
    recent = state.option_price_history[-n_time:]

    matrix = []
    for t in range(1, len(recent)):
        row = []
        valid = True
        for col in columns:
            prev_val = recent[t - 1].get(col, 0)
            curr_val = recent[t].get(col, 0)
            if prev_val > 0 and curr_val > 0:
                row.append((curr_val - prev_val) / prev_val)
            else:
                valid = False
                break
        if valid and len(row) == len(columns):
            matrix.append(row)

    if len(matrix) < 5 or len(columns) < 4:
        return {"explained_ratio": None, "direction": 0, "anomaly": False}

    M = np.array(matrix)
    M_centered = M - M.mean(axis=0)

    try:
        U, S_vals, Vt = np.linalg.svd(M_centered, full_matrices=False)
        total_var = np.sum(S_vals ** 2)
        if total_var == 0:
            return {"explained_ratio": None, "direction": 0, "anomaly": False}

        explained_ratio = (S_vals[0] ** 2) / total_var
        last_row_projection = U[-1, 0] * S_vals[0]
        direction = 1 if last_row_projection > 0 else (-1 if last_row_projection < 0 else 0)
        residual = np.sum(S_vals[1:] ** 2) / total_var
        anomaly = residual > 0.5

        state.pca_explained = float(explained_ratio)
        state.pca_direction = int(direction)

        return {
            "explained_ratio": float(explained_ratio),
            "direction": int(direction),
            "anomaly": anomaly
        }
    except:
        return {"explained_ratio": None, "direction": 0, "anomaly": False}


# ================================================================
# SECTION 6: IMPLIED PROBABILITY DISTRIBUTION (Breeden-Litzenberger)
# ================================================================

def implied_distribution(enriched_chain, nifty_price, T, r):
    sorted_strikes = sorted(enriched_chain.keys())
    if len(sorted_strikes) < 5:
        return None

    call_prices = {}
    for strike in sorted_strikes:
        if "CE" in enriched_chain[strike]:
            ltp = enriched_chain[strike]["CE"].get("kalman_price",
                                                   enriched_chain[strike]["CE"].get("ltp", 0))
            if ltp > 0:
                call_prices[strike] = ltp

    sorted_available = sorted(call_prices.keys())
    if len(sorted_available) < 5:
        return None

    delta_k = config.STRIKE_GAP
    density = {}

    for i in range(1, len(sorted_available) - 1):
        k_minus = sorted_available[i - 1]
        k = sorted_available[i]
        k_plus = sorted_available[i + 1]

        if k_plus - k_minus != 2 * delta_k:
            continue

        c_minus = call_prices[k_minus]
        c_mid = call_prices[k]
        c_plus = call_prices[k_plus]

        second_deriv = (c_minus - 2 * c_mid + c_plus) / (delta_k ** 2)
        prob_density = np.exp(r * T) * second_deriv

        if prob_density > 0:
            density[k] = prob_density

    if len(density) < 3:
        return None

    strikes_arr = np.array(sorted(density.keys()), dtype=float)
    probs = np.array([density[k] for k in sorted(density.keys())])
    total = np.sum(probs) * delta_k
    if total > 0:
        probs = probs / total

    mean = np.sum(strikes_arr * probs * delta_k)
    variance = np.sum((strikes_arr - mean) ** 2 * probs * delta_k)
    std = np.sqrt(max(variance, 0))
    third_moment = np.sum((strikes_arr - mean) ** 3 * probs * delta_k)
    skew = third_moment / (std ** 3) if std > 0 else 0

    return {
        "strikes": strikes_arr.tolist(),
        "probs": probs.tolist(),
        "mean": float(mean),
        "std": float(std),
        "skew": float(skew)
    }


def distribution_shift(current_dist, state):
    if current_dist is None:
        return {"mean_shift": 0, "std_change": 0, "skew": 0}

    mean_shift = 0
    std_change = 0

    if state.prev_dist_mean is not None:
        mean_shift = current_dist["mean"] - state.prev_dist_mean
    if state.prev_dist_std is not None and state.prev_dist_std > 0:
        std_change = (current_dist["std"] - state.prev_dist_std) / state.prev_dist_std

    state.prev_dist_mean = current_dist["mean"]
    state.prev_dist_std = current_dist["std"]
    state.prev_implied_dist = current_dist

    return {
        "mean_shift": float(mean_shift),
        "std_change": float(std_change),
        "skew": float(current_dist["skew"])
    }


# ================================================================
# SECTION 7: TRANSFER ENTROPY
# ================================================================

def calculate_transfer_entropy(state, atm_strike):
    n = len(state.nifty_return_bins)
    if n < config.TE_MIN_SNAPSHOTS:
        return {"leading_strikes": [], "max_te": 0, "direction": 0}

    nifty = state.nifty_return_bins
    results = []

    for key, opt_bins in state.option_return_bins.items():
        if len(opt_bins) < n:
            continue
        opt = opt_bins[-n:]
        nf = nifty[-n:]
        m = min(len(opt), len(nf))
        if m < config.TE_MIN_SNAPSHOTS:
            continue
        opt = opt[-m:]
        nf = nf[-m:]

        te = _compute_te(nf, opt)
        if te > 0.01:
            last_opt_dir = opt[-1]
            results.append({
                "key": key,
                "te": te,
                "direction": 1 if last_opt_dir == 2 else (-1 if last_opt_dir == 0 else 0)
            })

    results.sort(key=lambda x: -x["te"])

    if not results:
        return {"leading_strikes": [], "max_te": 0, "direction": 0}

    top = results[:3]
    avg_dir = np.mean([r["direction"] for r in top])
    direction = 1 if avg_dir > 0.3 else (-1 if avg_dir < -0.3 else 0)

    return {
        "leading_strikes": top,
        "max_te": top[0]["te"],
        "direction": direction
    }


def _compute_te(target, source):
    n = len(target)
    if n < 5:
        return 0

    joint_counts = {}
    cond_counts = {}
    total = 0

    for t in range(1, n):
        y_t = target[t]
        y_prev = target[t - 1]
        x_prev = source[t - 1]

        joint_key = (y_t, y_prev, x_prev)
        cond_key = (y_t, y_prev)

        joint_counts[joint_key] = joint_counts.get(joint_key, 0) + 1
        cond_counts[cond_key] = cond_counts.get(cond_key, 0) + 1
        total += 1

    if total == 0:
        return 0

    margin_yx = {}
    margin_y_given_x = {}

    for (y, yp, xp), count in joint_counts.items():
        margin_yx[yp] = margin_yx.get(yp, 0) + count
        key_ypxp = (yp, xp)
        margin_y_given_x[key_ypxp] = margin_y_given_x.get(key_ypxp, 0) + count

    te = 0
    for (y, yp, xp), count in joint_counts.items():
        p_joint = count / total
        p_y_given_yp = cond_counts.get((y, yp), 0) / margin_yx.get(yp, 1)
        p_y_given_ypxp = count / margin_y_given_x.get((yp, xp), 1)

        if p_y_given_yp > 0 and p_y_given_ypxp > 0:
            te += p_joint * np.log2(p_y_given_ypxp / p_y_given_yp)

    return max(0, te)


# ================================================================
# SECTION 8: VOTER 1 — PATTERN
# ================================================================

def voter_pattern(nifty_closes, pattern_store, state):
    if len(nifty_closes) < 10 or pattern_store is None:
        return 50.0
    n = min(len(nifty_closes), 20)
    current = np.array(nifty_closes[-n:], dtype=float)
    current_ret = np.diff(current) / current[:-1]

    results = pattern_store.find_matches(
        current_ret,
        current_regime=state.current_regime
    )
    if results is None or results["total_matches"] < config.MIN_PATTERN_MATCHES:
        return 50.0

    up_pct = results["weighted_up"] / results["total_weight"] * 100
    return max(30.0, min(70.0, up_pct))


# ================================================================
# SECTION 9: VOTER 2 — MOMENTUM (multi-lag, significance)
# ================================================================

def voter_momentum(nifty_closes):
    if len(nifty_closes) < 20:
        return 50.0

    prices = np.array(nifty_closes[-30:] if len(nifty_closes) >= 30 else nifty_closes, dtype=float)
    returns = np.diff(prices) / prices[:-1]
    n = len(returns)
    if n < 10:
        return 50.0

    mean_r = np.mean(returns)
    demeaned = returns - mean_r
    var = np.sum(demeaned ** 2)
    if var == 0:
        return 50.0

    significance_threshold = 1.96 / np.sqrt(n)
    lags = [1, 2, 3, 5]
    significant_autocorrs = []

    for lag in lags:
        if lag >= n - 1:
            continue
        acf = np.sum(demeaned[:-lag] * demeaned[lag:]) / var
        if abs(acf) > significance_threshold:
            significant_autocorrs.append(acf)

    if not significant_autocorrs:
        return 50.0

    avg_acf = np.mean(significant_autocorrs)
    recent_10 = np.sum(returns[-10:]) if n >= 10 else np.sum(returns[-5:])

    if recent_10 > 0:
        prob = 50 + min(avg_acf * 35, 20) if avg_acf > 0 else 50 + max(avg_acf * 25, -15)
    else:
        prob = 50 - min(abs(avg_acf) * 35, 20) if avg_acf > 0 else 50 + min(abs(avg_acf) * 25, 15)

    return max(25.0, min(75.0, prob))


# ================================================================
# SECTION 10: VOTER 3 — CYCLE (fixed phase, windowed FFT)
# ================================================================

def voter_cycle(nifty_closes):
    if len(nifty_closes) < 30:
        return 50.0

    prices = np.array(nifty_closes[-60:] if len(nifty_closes) >= 60 else nifty_closes[-30:], dtype=float)
    returns = np.diff(prices) / prices[:-1]
    n = len(returns)
    if n < 15:
        return 50.0

    returns_detrended = returns - np.mean(returns)
    window = np.hanning(n)
    windowed = returns_detrended * window

    fft_result = fft(windowed)
    spectrum = np.abs(fft_result[:n // 2])
    phases = np.angle(fft_result[:n // 2])

    if len(spectrum) < 4:
        return 50.0

    spectrum[0] = 0
    if len(spectrum) > 1:
        spectrum[1] = 0

    noise_floor = np.mean(spectrum[2:]) if len(spectrum) > 2 else 0
    dominant_idx = np.argmax(spectrum)

    if dominant_idx == 0 or spectrum[dominant_idx] < noise_floor * 2.5:
        return 50.0

    phase = phases[dominant_idx]
    phase_normalized = (phase + np.pi) / (2 * np.pi)
    signal_strength = spectrum[dominant_idx] / noise_floor if noise_floor > 0 else 1

    if phase_normalized < 0.25:
        prob = 50 + (0.25 - phase_normalized) * 60
    elif phase_normalized < 0.5:
        prob = 50 + (0.5 - phase_normalized) * 40
    elif phase_normalized < 0.75:
        prob = 50 - (phase_normalized - 0.5) * 60
    else:
        prob = 50 - (0.25 - (phase_normalized - 0.75)) * 40

    strength_scale = min(signal_strength / 3.0, 1.5)
    prob = 50 + (prob - 50) * strength_scale

    return max(35.0, min(65.0, prob))


# ================================================================
# SECTION 11: VOTER 4 — HURST
# ================================================================

def voter_hurst(nifty_closes, state):
    if len(nifty_closes) < 60:
        return 50.0

    prices = np.array(nifty_closes[-200:] if len(nifty_closes) >= 200 else nifty_closes[-60:], dtype=float)
    returns = np.diff(np.log(prices))
    n = len(returns)
    if n < 30:
        return 50.0

    max_k = min(n // 2, 80)
    rs_values = []
    sizes = []

    for k in range(15, max_k, 5):
        rs_list = []
        for start in range(0, n - k, k):
            chunk = returns[start:start + k]
            if len(chunk) < k:
                continue
            mean_c = np.mean(chunk)
            cumdev = np.cumsum(chunk - mean_c)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(chunk, ddof=1)
            if S > 1e-10:
                rs_list.append(R / S)
        if len(rs_list) >= 2:
            rs_values.append(np.mean(rs_list))
            sizes.append(k)

    if len(rs_values) < 3:
        state.last_hurst = 0.5
        return 50.0

    log_sizes = np.log(sizes)
    log_rs = np.log(rs_values)
    hurst = np.polyfit(log_sizes, log_rs, 1)[0]
    hurst = max(0.0, min(1.0, hurst))
    state.last_hurst = hurst

    recent_direction = np.sum(returns[-15:])

    if hurst > 0.55:
        strength = (hurst - 0.5) * 40
        prob = (50 + strength) if recent_direction > 0 else (50 - strength)
    elif hurst < 0.45:
        strength = (0.5 - hurst) * 30
        prob = (50 - strength) if recent_direction > 0 else (50 + strength)
    else:
        prob = 50.0

    return max(30.0, min(70.0, prob))


# ================================================================
# SECTION 12: VOTER 5 — OI/PCR
# ================================================================

def voter_oi(enriched_chain, atm_strike):
    total_ce_oi = 0
    total_pe_oi = 0
    max_ce_oi = 0
    max_ce_strike = atm_strike
    max_pe_oi = 0
    max_pe_strike = atm_strike
    ce_vol_total = 0
    pe_vol_total = 0

    for strike, types in enriched_chain.items():
        if "CE" in types:
            oi = types["CE"].get("oi", 0)
            total_ce_oi += oi
            ce_vol_total += types["CE"].get("volume", 0)
            if oi > max_ce_oi:
                max_ce_oi = oi
                max_ce_strike = strike
        if "PE" in types:
            oi = types["PE"].get("oi", 0)
            total_pe_oi += oi
            pe_vol_total += types["PE"].get("volume", 0)
            if oi > max_pe_oi:
                max_pe_oi = oi
                max_pe_strike = strike

    prob = 50.0

    if total_ce_oi > 0:
        pcr = total_pe_oi / total_ce_oi
        if pcr > 1.3:
            prob += 10
        elif pcr > 1.1:
            prob += 5
        elif pcr < 0.7:
            prob -= 10
        elif pcr < 0.9:
            prob -= 5

    if max_ce_strike > atm_strike and max_pe_strike < atm_strike:
        dist_up = max_ce_strike - atm_strike
        dist_down = atm_strike - max_pe_strike
        total = dist_up + dist_down
        if total > 0:
            room_up_ratio = dist_up / total
            if room_up_ratio > 0.6:
                prob += 5
            elif room_up_ratio < 0.4:
                prob -= 5

    if ce_vol_total > 0 and pe_vol_total > 0:
        vol_ratio = pe_vol_total / ce_vol_total
        if vol_ratio > 1.5:
            prob += 5
        elif vol_ratio < 0.67:
            prob -= 5

    return max(30.0, min(70.0, prob))


# ================================================================
# SECTION 13: VOTER 6 — IV + IMPLIED DISTRIBUTION (FIXED)
# ================================================================

def voter_iv_distribution(enriched_chain, atm_strike, state, days_to_expiry):
    prob = 50.0

    # FIX: use actual nifty spot stored in state, not a strike
    nifty_price = state.nifty_spot if state.nifty_spot > 0 else atm_strike

    dist = implied_distribution(enriched_chain, nifty_price, days_to_expiry, config.RISK_FREE_RATE)

    if dist is not None:
        shift = distribution_shift(dist, state)

        if shift["mean_shift"] > 5:
            prob += 8
        elif shift["mean_shift"] > 2:
            prob += 4
        elif shift["mean_shift"] < -5:
            prob -= 8
        elif shift["mean_shift"] < -2:
            prob -= 4

        if shift["skew"] > 0.3:
            prob += 4
        elif shift["skew"] < -0.3:
            prob -= 4

    # IV skew analysis
    ce_ivs = []
    pe_ivs = []
    for check_strike in range(atm_strike - 150, atm_strike + 200, 50):
        if check_strike in enriched_chain:
            types = enriched_chain[check_strike]
            if "CE" in types and types["CE"].get("iv") is not None:
                ce_ivs.append(types["CE"]["iv"])
            if "PE" in types and types["PE"].get("iv") is not None:
                pe_ivs.append(types["PE"]["iv"])

    if ce_ivs and pe_ivs:
        iv_skew = np.mean(pe_ivs) - np.mean(ce_ivs)
        if iv_skew > 0.05:
            prob += 3
        elif iv_skew < -0.03:
            prob -= 3

    return max(30.0, min(70.0, prob))


# ================================================================
# SECTION 14: VOTER 7 — SMART MONEY
# ================================================================

def voter_smart_money(enriched_chain, atm_strike, option_snapshots, state):
    prob = 50.0
    signals = []

    near_strikes = [s for s in enriched_chain.keys() if abs(s - atm_strike) <= 250]

    ce_oi_near = 0
    pe_oi_near = 0
    total_ce_buy = 0
    total_ce_sell = 0
    total_pe_buy = 0
    total_pe_sell = 0

    for strike in near_strikes:
        types = enriched_chain[strike]
        if "CE" in types:
            ce_oi_near += types["CE"].get("oi", 0)
            total_ce_buy += types["CE"].get("buy_qty", 0)
            total_ce_sell += types["CE"].get("sell_qty", 0)
        if "PE" in types:
            pe_oi_near += types["PE"].get("oi", 0)
            total_pe_buy += types["PE"].get("buy_qty", 0)
            total_pe_sell += types["PE"].get("sell_qty", 0)

    if ce_oi_near > 0 and pe_oi_near > 0:
        oi_ratio = pe_oi_near / ce_oi_near
        if oi_ratio > 1.3:
            signals.append(8)
        elif oi_ratio > 1.1:
            signals.append(4)
        elif oi_ratio < 0.7:
            signals.append(-8)
        elif oi_ratio < 0.9:
            signals.append(-4)

    if total_ce_buy > 0 and total_ce_sell > 0:
        ce_imb = total_ce_buy / total_ce_sell
        if ce_imb > 1.5:
            signals.append(6)
        elif ce_imb < 0.67:
            signals.append(-4)

    if total_pe_buy > 0 and total_pe_sell > 0:
        pe_imb = total_pe_buy / total_pe_sell
        if pe_imb > 1.5:
            signals.append(-6)
        elif pe_imb < 0.67:
            signals.append(4)

    # unusual volume
    all_vols = []
    vol_map = {}
    for strike in near_strikes:
        for opt_type in ["CE", "PE"]:
            if opt_type in enriched_chain.get(strike, {}):
                vol = enriched_chain[strike][opt_type].get("volume", 0)
                if vol > 0:
                    all_vols.append(vol)
                    vol_map[(strike, opt_type)] = vol

    if len(all_vols) > 5:
        avg_v = np.mean(all_vols)
        std_v = np.std(all_vols)
        if std_v > 0:
            for (strike, opt_type), vol in vol_map.items():
                z = (vol - avg_v) / std_v
                if z > 2.0:
                    if opt_type == "CE" and strike > atm_strike:
                        signals.append(4)
                    elif opt_type == "PE" and strike < atm_strike:
                        signals.append(-4)

    # OI change
    if len(option_snapshots) >= 3:
        prev_opts = option_snapshots[-3].get("options", {})
        ce_oi_change = 0
        pe_oi_change = 0
        for strike in near_strikes:
            if strike in enriched_chain and strike in prev_opts:
                curr_ce = enriched_chain[strike].get("CE", {}).get("oi", 0)
                prev_ce = prev_opts.get(strike, {}).get("CE", {}).get("oi", 0) if isinstance(prev_opts.get(strike), dict) else 0
                ce_oi_change += (curr_ce - prev_ce) if curr_ce > 0 and prev_ce > 0 else 0

                curr_pe = enriched_chain[strike].get("PE", {}).get("oi", 0)
                prev_pe = prev_opts.get(strike, {}).get("PE", {}).get("oi", 0) if isinstance(prev_opts.get(strike), dict) else 0
                pe_oi_change += (curr_pe - prev_pe) if curr_pe > 0 and prev_pe > 0 else 0

        if pe_oi_change > ce_oi_change * 1.5 and pe_oi_change > 500:
            signals.append(7)
        elif ce_oi_change > pe_oi_change * 1.5 and ce_oi_change > 500:
            signals.append(-7)

    # Transfer Entropy
    te_result = calculate_transfer_entropy(state, atm_strike)
    if te_result["max_te"] > 0.02:
        te_dir = te_result["direction"]
        te_strength = min(te_result["max_te"] * 200, 10)
        signals.append(int(te_dir * te_strength))

    if not signals:
        return 50.0

    total_signal = sum(signals)
    prob = 50.0 + max(-20, min(20, total_signal))
    return max(30.0, min(70.0, prob))


# ================================================================
# SECTION 15: VOTER 8 — REGIME (EMA smoothed)
# ================================================================

def voter_regime(nifty_closes, state):
    if len(nifty_closes) < 20:
        return "unknown", {}, {}

    prices = np.array(nifty_closes[-40:] if len(nifty_closes) >= 40 else nifty_closes, dtype=float)
    returns = np.diff(prices) / prices[:-1]
    n = len(returns)
    if n < 10:
        return "unknown", {}, {}

    vol = float(np.std(returns))
    mean_ret = float(np.mean(returns))
    trend_strength = abs(mean_ret) / vol if vol > 0 else 0
    recent_vol = float(np.std(returns[-10:])) if n >= 10 else vol
    vol_ratio = recent_vol / vol if vol > 0 else 1

    raw_scores = {"trending_calm": 0, "trending_volatile": 0,
                  "ranging_calm": 0, "ranging_volatile": 0}

    if trend_strength > 0.3:
        if vol_ratio < 1.2:
            raw_scores["trending_calm"] = trend_strength * 2
        else:
            raw_scores["trending_volatile"] = trend_strength * 1.5
    else:
        if vol_ratio < 0.8:
            raw_scores["ranging_calm"] = (0.5 - trend_strength) * 2
        else:
            raw_scores["ranging_volatile"] = vol_ratio

    total_raw = sum(raw_scores.values())
    if total_raw > 0:
        for k in raw_scores:
            raw_scores[k] /= total_raw
    else:
        for k in raw_scores:
            raw_scores[k] = 0.25

    alpha = config.REGIME_SMOOTHING_ALPHA
    for k in state.regime_scores:
        state.regime_scores[k] = alpha * raw_scores.get(k, 0) + (1 - alpha) * state.regime_scores[k]

    total_smooth = sum(state.regime_scores.values())
    if total_smooth > 0:
        for k in state.regime_scores:
            state.regime_scores[k] /= total_smooth

    regime = max(state.regime_scores, key=state.regime_scores.get)
    state.current_regime = regime

    adjustments = {
        "trending_calm": {"momentum": 1.5, "hurst": 1.3, "cycle": 0.5},
        "trending_volatile": {"momentum": 1.3, "hurst": 1.2, "oi": 1.3, "smart_money": 1.3},
        "ranging_calm": {"cycle": 1.5, "momentum": 0.5, "oi": 1.3},
        "ranging_volatile": {"momentum": 0.5, "cycle": 0.3, "oi": 1.5, "smart_money": 1.5}
    }.get(regime, {})

    return regime, adjustments, dict(state.regime_scores)


# ================================================================
# SECTION 16: VOTER 9 — ENTROPY (fixed + chain)
# ================================================================

def voter_entropy(nifty_closes, enriched_chain=None):
    if len(nifty_closes) < 20:
        return 1.0

    prices = np.array(nifty_closes[-50:] if len(nifty_closes) >= 50 else nifty_closes[-20:], dtype=float)
    returns = np.diff(prices) / prices[:-1]
    n = len(returns)
    if n < 15:
        return 1.0

    n_bins = max(5, min(int(np.sqrt(n)), 15))
    hist, _ = np.histogram(returns, bins=n_bins)
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    if len(probs) < 2:
        return 1.0

    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(n_bins)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

    chain_factor = 1.0
    if enriched_chain:
        ivs = []
        for strike, types in enriched_chain.items():
            for opt_type in ["CE", "PE"]:
                if opt_type in types and types[opt_type].get("iv") is not None:
                    ivs.append(types[opt_type]["iv"])
        if len(ivs) > 5:
            iv_cv = np.std(ivs) / np.mean(ivs) if np.mean(ivs) > 0 else 0
            if iv_cv > 0.3:
                chain_factor = 0.85
            elif iv_cv < 0.1:
                chain_factor = 1.1

    if norm_entropy < 0.4:
        confidence = 1.0
    elif norm_entropy > 0.75:
        confidence = 0.45
    else:
        confidence = 1.0 - (norm_entropy - 0.4) * 1.57

    confidence *= chain_factor
    return max(0.35, min(1.0, confidence))


# ================================================================
# SECTION 17: MAIN PROBABILITY ENGINE
# ================================================================

def calculate_probability(nifty_closes, enriched_chain, atm_strike,
                          option_snapshots, prev_enriched, pattern_store,
                          state, days_to_expiry):
    state.increment()
    state.store_option_prices(enriched_chain, nifty_closes[-1] if nifty_closes else 0)
    state.store_binned_returns(nifty_closes, enriched_chain, atm_strike)

    pca = pca_chain_analysis(state, enriched_chain, atm_strike)

    v1 = voter_pattern(nifty_closes, pattern_store, state)
    v2 = voter_momentum(nifty_closes)
    v3 = voter_cycle(nifty_closes)
    v4 = voter_hurst(nifty_closes, state)
    v5 = voter_oi(enriched_chain, atm_strike)
    v6 = voter_iv_distribution(enriched_chain, atm_strike, state, days_to_expiry)
    v7 = voter_smart_money(enriched_chain, atm_strike, option_snapshots, state)

    regime, weight_adj, regime_probs = voter_regime(nifty_closes, state)
    confidence = voter_entropy(nifty_closes, enriched_chain)

    if pca["explained_ratio"] is not None:
        if pca["explained_ratio"] > 0.7:
            confidence *= 1.15
        elif pca["explained_ratio"] < 0.3:
            confidence *= 0.85
    confidence = max(0.35, min(1.0, confidence))

    votes = {
        "pattern": v1, "momentum": v2, "cycle": v3, "hurst": v4,
        "oi": v5, "iv_dist": v6, "smart_money": v7
    }

    weights = dict(config.VOTER_WEIGHTS)
    for voter_name, adj in weight_adj.items():
        if voter_name in weights:
            weights[voter_name] *= adj

    total_weight = 0
    weighted_sum = 0
    voter_details = {}

    for name, vote in votes.items():
        w = weights.get(name, 1.0)
        weighted_sum += vote * w
        total_weight += w
        voter_details[name] = {"vote": round(vote, 1), "weight": round(w, 2)}

    raw_prob = weighted_sum / total_weight if total_weight > 0 else 50.0
    final_prob_up = 50 + (raw_prob - 50) * confidence
    final_prob_up = max(20.0, min(80.0, final_prob_up))
    final_prob_down = 100 - final_prob_up

    deviation = abs(final_prob_up - 50)
    strength = "STRONG" if deviation > 15 else ("MODERATE" if deviation > 8 else
                                                ("WEAK" if deviation > 3 else "NEUTRAL"))

    exit_signals = check_exit_signals(state, final_prob_up, regime)

    return {
        "prob_up": round(final_prob_up, 1),
        "prob_down": round(final_prob_down, 1),
        "strength": strength,
        "regime": regime,
        "regime_probs": regime_probs,
        "confidence": round(confidence, 2),
        "voters": voter_details,
        "hurst": round(state.last_hurst, 3),
        "pca": pca,
        "exit_signals": exit_signals
    }


# ================================================================
# SECTION 18: EDGE CALCULATION (Hurst-adjusted)
# ================================================================

def calculate_edge(enriched_chain, nifty_price, prob_up, days_to_expiry, hurst=0.5):
    r = config.RISK_FREE_RATE
    T = days_to_expiry
    hurst_factor = 1.0 + (hurst - 0.5) * 0.8
    edges = {}

    for strike, types in enriched_chain.items():
        edges[strike] = {}
        for opt_type in ["CE", "PE"]:
            if opt_type not in types:
                continue
            data = types[opt_type]
            market_price = data["ltp"]
            iv = data.get("iv")

            if market_price <= 0 or iv is None or iv <= 0:
                edges[strike][opt_type] = None
                continue

            adjusted_iv = iv * hurst_factor
            our_price = bs_price(nifty_price, strike, T, r, adjusted_iv, opt_type)

            delta = abs(data.get("delta", 0.5))
            if opt_type == "CE":
                our_prob = prob_up / 100.0
                market_prob = delta
            else:
                our_prob = (100 - prob_up) / 100.0
                market_prob = delta

            prob_adjustment = (our_prob - market_prob) / market_prob if market_prob > 0.01 else 0

            bs_edge = our_price - market_price
            prob_edge = prob_adjustment * market_price * 0.3
            total_edge = bs_edge * 0.6 + prob_edge * 0.4

            edges[strike][opt_type] = round(total_edge, 1)

    return edges


# ================================================================
# SECTION 19: STRIKE OPTIMIZATION (with per-strike probability)
# ================================================================

def optimize_strike(enriched_chain, edges, prob_up, atm_strike, days_to_expiry):
    candidates = []

    for strike, types in enriched_chain.items():
        for opt_type in ["CE", "PE"]:
            if opt_type not in types:
                continue

            data = types[opt_type]
            edge = edges.get(strike, {}).get(opt_type)

            if edge is None or edge <= 0:
                continue

            ltp = data["ltp"]
            delta = abs(data.get("delta", 0))
            spread = data.get("spread", 999)
            volume = data.get("volume", 0)
            theta = abs(data.get("theta", 0))

            if ltp <= 1 or volume < 50:
                continue

            spread_pct = spread / ltp if ltp > 0 else 1
            if spread_pct > 0.05:
                continue

            if edge < spread * 1.5:
                continue

            # per-strike probability
            if opt_type == "CE":
                strike_prob = delta * (prob_up / 50.0)
                if strike_prob < 0.60:
                    continue
            else:
                strike_prob = delta * ((100 - prob_up) / 50.0)
                if strike_prob < 0.60:
                    continue

            # scoring
            edge_score = min(edge / max(ltp, 1) * 10, 3.0)
            liquidity_score = min(volume / 5000, 1.0)
            spread_score = max(0, 1 - spread_pct * 20)
            delta_score = min(delta / 0.3, 1.0)
            prob_score = max(0, min(1.5, (strike_prob - 0.5) * 3))

            theta_score = 1.0
            if days_to_expiry < 0.005:
                theta_cost = theta * 2 / max(ltp, 1)
                theta_score = max(0, 1 - theta_cost * 5)

            total = (edge_score * 3.0 + liquidity_score * 2.0 + spread_score * 1.5 +
                     delta_score * 1.0 + theta_score * 1.0 + prob_score * 2.0)

            strike_prob_pct = strike_prob * 100

            if opt_type == "CE":
                if strike_prob_pct >= 75:
                    signal = "STRONG BUY CE"
                elif strike_prob_pct >= 65:
                    signal = "BUY CE"
                elif strike_prob_pct >= 60:
                    signal = "watch CE"
                else:
                    signal = ""
            else:
                if strike_prob_pct >= 75:
                    signal = "STRONG BUY PE"
                elif strike_prob_pct >= 65:
                    signal = "BUY PE"
                elif strike_prob_pct >= 60:
                    signal = "watch PE"
                else:
                    signal = ""

            candidates.append({
                "strike": strike,
                "type": opt_type,
                "score": round(total, 2),
                "edge": edge,
                "ltp": ltp,
                "delta": round(delta, 3),
                "spread": spread,
                "volume": volume,
                "signal": signal,
                "strike_prob": round(strike_prob_pct, 1)
            })

    candidates.sort(key=lambda x: -x["score"])
    return candidates[:5]


# ================================================================
# SECTION 20: EXIT SIGNALS
# ================================================================

def check_exit_signals(state, current_prob_up, current_regime):
    exits = []
    new_active = []

    for signal in state.active_signals:
        candles_held = state.candle_count - signal["entry_candle"]
        entry_direction = signal["direction"]
        entry_prob = signal["entry_prob"]

        if entry_direction == "bullish" and current_prob_up < 45:
            exits.append({
                "type": "REVERSAL",
                "message": f"Was bullish {entry_prob}%, now {current_prob_up}% bearish",
                "candles_held": candles_held
            })
            continue

        if entry_direction == "bearish" and current_prob_up > 55:
            exits.append({
                "type": "REVERSAL",
                "message": f"Was bearish {entry_prob}%, now {current_prob_up}% bullish",
                "candles_held": candles_held
            })
            continue

        if candles_held > 20:
            exits.append({
                "type": "TIME",
                "message": f"Held {candles_held * 3} min — consider exit",
                "candles_held": candles_held
            })
            continue

        if signal.get("entry_regime") != current_regime and candles_held > 5:
            exits.append({
                "type": "REGIME",
                "message": f"{signal.get('entry_regime')} → {current_regime}",
                "candles_held": candles_held
            })
            continue

        if entry_direction == "bullish" and current_prob_up < 52:
            exits.append({
                "type": "WEAK",
                "message": f"Bullish weakened to {current_prob_up}%",
                "candles_held": candles_held
            })
            continue

        if entry_direction == "bearish" and current_prob_up > 48:
            exits.append({
                "type": "WEAK",
                "message": f"Bearish weakened to {current_prob_up}%",
                "candles_held": candles_held
            })
            continue

        new_active.append(signal)

    if current_prob_up > 58 or current_prob_up < 42:
        direction = "bullish" if current_prob_up > 58 else "bearish"
        already = any(s["direction"] == direction for s in new_active)
        if not already:
            new_active.append({
                "direction": direction,
                "entry_prob": current_prob_up,
                "entry_candle": state.candle_count,
                "entry_regime": current_regime
            })

    state.active_signals = new_active
    return exits