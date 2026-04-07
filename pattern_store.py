import os
import json
import numpy as np
from datetime import datetime, timedelta
import config


class PatternStore:
    def __init__(self):
        self.patterns = []
        self.daily_records = []
        self._save_counter = 0
        self._candle_counter = 0
        self.load()

    def load(self):
        if os.path.exists(config.PATTERN_FILE):
            try:
                with open(config.PATTERN_FILE, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self.patterns = data.get("patterns", [])
                    self.daily_records = data.get("daily_records", [])
                elif isinstance(data, list):
                    self.patterns = data
                    self.daily_records = []
                print(f"📚 Loaded {len(self.patterns)} patterns, "
                      f"{len(self.daily_records)} daily records")
            except:
                self.patterns = []
                self.daily_records = []
                print("📚 Fresh start")
        else:
            self.patterns = []
            self.daily_records = []
            print("📚 First run — empty library")

    def save(self):
        data = {"patterns": self.patterns, "daily_records": self.daily_records}
        try:
            with open(config.PATTERN_FILE, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"   ⚠️ Save failed: {e}")

    def _auto_save(self):
        self._save_counter += 1
        if self._save_counter >= 5:
            self.save()
            self._save_counter = 0

    def store_candle_data(self, nifty_closes, prob_result, snapshot):
        if len(nifty_closes) < 3:
            return

        # only store every Nth candle to avoid duplicates
        self._candle_counter += 1
        if self._candle_counter % config.PATTERN_STEP != 0:
            return

        n = min(len(nifty_closes), 20)
        closes_arr = np.array(nifty_closes[-n:], dtype=float)
        if len(closes_arr) < 5:
            return

        returns = np.diff(closes_arr) / closes_arr[:-1]

        # compute extra features for multi-representation matching
        vol = float(np.std(returns))
        mean_ret = float(np.mean(returns))
        trend = float(np.sum(returns[-5:])) if len(returns) >= 5 else 0

        entry = {
            "timestamp": datetime.now().isoformat(),
            "nifty_price": float(nifty_closes[-1]),
            "pattern_returns": [float(r) for r in returns],
            "volatility": vol,
            "mean_return": mean_ret,
            "recent_trend": trend,
            "prob_up": prob_result["prob_up"],
            "regime": prob_result.get("regime", "unknown"),
            "result_5": None,
            "result_10": None,
            "result_20": None,
            "result_5_magnitude": None,
            "source": "live"
        }

        self.daily_records.append(entry)
        self._auto_save()

    def fill_results(self, nifty_closes):
        if len(nifty_closes) < 2:
            return

        current_price = float(nifty_closes[-1])
        updated = False

        for i, record in enumerate(self.daily_records):
            candles_ago = len(self.daily_records) - 1 - i
            ref_price = record["nifty_price"]
            if ref_price <= 0:
                continue

            move = current_price - ref_price

            if record["result_5"] is None and candles_ago >= 5:
                # minimum move threshold
                if abs(move) >= config.PATTERN_MIN_MOVE:
                    record["result_5"] = "up" if move > 0 else "down"
                else:
                    record["result_5"] = "flat"
                record["result_5_magnitude"] = float(move)
                updated = True

                # promote to patterns
                if record["result_5"] != "flat":
                    self.patterns.append(record.copy())

            if record["result_10"] is None and candles_ago >= 10:
                if abs(move) >= config.PATTERN_MIN_MOVE:
                    record["result_10"] = "up" if move > 0 else "down"
                else:
                    record["result_10"] = "flat"
                updated = True

            if record["result_20"] is None and candles_ago >= 20:
                if abs(move) >= config.PATTERN_MIN_MOVE:
                    record["result_20"] = "up" if move > 0 else "down"
                else:
                    record["result_20"] = "flat"
                updated = True

        if updated:
            self._auto_save()

    def find_matches(self, current_returns, current_regime=None):
        if len(self.patterns) < config.MIN_PATTERN_MATCHES:
            return None

        current = np.array(current_returns, dtype=float)
        if len(current) < 5:
            return None

        curr_vol = float(np.std(current))
        curr_trend = float(np.sum(current[-5:])) if len(current) >= 5 else 0

        matches = []
        now = datetime.now()

        for pattern in self.patterns:
            # skip flat results
            r5 = pattern.get("result_5")
            if r5 is None or r5 == "flat":
                continue

            stored_raw = pattern.get("pattern_returns", [])
            if len(stored_raw) < 5:
                continue

            stored = np.array(stored_raw, dtype=float)
            min_len = min(len(current), len(stored))
            if min_len < 5:
                continue

            c = current[-min_len:]
            s = stored[-min_len:]

            if np.std(c) < 1e-10 or np.std(s) < 1e-10:
                continue

            corr = np.corrcoef(c, s)[0, 1]
            if np.isnan(corr):
                continue

            if corr < config.PATTERN_MATCH_THRESHOLD:
                continue

            # multi-representation: volatility similarity
            stored_vol = pattern.get("volatility", 0)
            if stored_vol > 0 and curr_vol > 0:
                vol_ratio = min(curr_vol, stored_vol) / max(curr_vol, stored_vol)
                if vol_ratio < 0.3:
                    continue  # very different volatility — skip

            # recency weight
            try:
                ts = datetime.fromisoformat(pattern["timestamp"].replace('+05:30', ''))
                days_ago = (now - ts).days
                recency_weight = max(0.3, 1.0 - days_ago / config.PATTERN_DECAY_DAYS)
            except:
                recency_weight = 0.5

            # regime bonus
            regime_weight = 1.0
            if current_regime and pattern.get("regime") == current_regime:
                regime_weight = 1.5
            elif current_regime and pattern.get("regime") != current_regime:
                regime_weight = 0.7

            # correlation bonus
            corr_weight = (corr - config.PATTERN_MATCH_THRESHOLD) / (1 - config.PATTERN_MATCH_THRESHOLD)

            total_weight = recency_weight * regime_weight * (0.5 + corr_weight)

            matches.append({
                "result": r5,
                "weight": total_weight,
                "magnitude": pattern.get("result_5_magnitude", 0)
            })

        if len(matches) < config.MIN_PATTERN_MATCHES:
            return None

        total_weight = sum(m["weight"] for m in matches)
        weighted_up = sum(m["weight"] for m in matches if m["result"] == "up")
        avg_magnitude = np.mean([m["magnitude"] for m in matches if m["magnitude"] is not None and m["magnitude"] != 0])

        return {
            "total_matches": len(matches),
            "total_weight": total_weight,
            "weighted_up": weighted_up,
            "weighted_down": total_weight - weighted_up,
            "avg_magnitude": float(avg_magnitude) if not np.isnan(avg_magnitude) else 0,
            "up_count": sum(1 for m in matches if m["result"] == "up"),
            "down_count": sum(1 for m in matches if m["result"] == "down")
        }

    def end_of_day_update(self):
        promoted = 0
        for record in self.daily_records:
            if record.get("result_5") is not None and record["result_5"] != "flat":
                already = any(p.get("timestamp") == record["timestamp"] for p in self.patterns)
                if not already:
                    self.patterns.append(record)
                    promoted += 1

        cutoff = datetime.now() - timedelta(days=config.PATTERN_DECAY_DAYS)
        before = len(self.patterns)
        self.patterns = [p for p in self.patterns
                         if datetime.fromisoformat(p["timestamp"].replace('+05:30', '')) > cutoff]
        pruned = before - len(self.patterns)

        # deduplicate
        seen = set()
        unique = []
        for p in self.patterns:
            ts = p.get("timestamp", "")
            if ts not in seen:
                seen.add(ts)
                unique.append(p)
        self.patterns = unique

        # memory check
        data_str = json.dumps(self.patterns)
        size_mb = len(data_str.encode()) / (1024 * 1024)
        if size_mb > config.MAX_MEMORY_MB:
            self.patterns.sort(key=lambda x: x["timestamp"], reverse=True)
            while len(self.patterns) > 100:
                if len(json.dumps(self.patterns).encode()) / (1024 * 1024) <= config.MAX_MEMORY_MB * 0.8:
                    break
                self.patterns.pop()

        self.daily_records = []
        self.save()

        final_size = len(json.dumps(self.patterns).encode()) / (1024 * 1024)
        return {
            "total_patterns": len(self.patterns),
            "added_today": promoted,
            "pruned": pruned,
            "memory_mb": round(final_size, 1)
        }

    def get_stats(self):
        try:
            data = {"patterns": self.patterns, "daily_records": self.daily_records}
            size_mb = len(json.dumps(data).encode()) / (1024 * 1024)
        except:
            size_mb = 0
        return {
            "total_patterns": len(self.patterns),
            "memory_mb": round(size_mb, 1),
            "today_records": len(self.daily_records)
        }