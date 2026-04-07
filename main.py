import os
import sys
import time
import json
from datetime import datetime

import config
from data_fetcher import DataFetcher
from math_brain import (AnalysisState, enrich_chain, calculate_probability,
                        calculate_edge, optimize_strike)
from pattern_store import PatternStore
from excel_dashboard import LiveDashboard


def print_banner():
    print("\n" + "=" * 60)
    print("   NIFTY OPTIONS INTELLIGENCE SYSTEM v2")
    print("   All 9 Math Concepts Implemented")
    print("=" * 60)


def print_signal(prob_result, nifty_price, atm, best_picks):
    p = prob_result
    d = "⬆ UP" if p["prob_up"] > 55 else ("⬇ DOWN" if p["prob_up"] < 45 else "↔ FLAT")

    print(f"\n{'─' * 55}")
    print(f"  NIFTY: {nifty_price:.0f}  |  ATM: {atm}  |  Hurst: {p.get('hurst', 0.5):.2f}")
    print(f"  {d}  |  Up: {p['prob_up']}%  Down: {p['prob_down']}%")
    print(f"  Strength: {p['strength']}  |  Regime: {p['regime']}  |  Conf: {p['confidence']}")

    voters = p.get("voters", {})
    print(f"  Voters: " + " | ".join(f"{k}:{v['vote']:.0f}%" for k, v in voters.items()))

    rp = p.get("regime_probs", {})
    if rp:
        print(f"  Regime probs: " + " | ".join(f"{k}:{v:.0%}" for k, v in rp.items()))

    pca = p.get("pca", {})
    if pca.get("explained_ratio"):
        print(f"  PCA: {pca['explained_ratio']:.0%} explained | dir: {pca['direction']}")

    exits = p.get("exit_signals", [])
    for ex in exits:
        print(f"  ⚠ EXIT: {ex['type']} — {ex['message']}")

    if best_picks:
        print(f"  📌 TOP PICK: {best_picks[0]['strike']} {best_picks[0]['type']} "
              f"(edge: {best_picks[0]['edge']:+.1f}, score: {best_picks[0]['score']})")

    print(f"{'─' * 55}")


def main():
    print_banner()
    os.makedirs(config.DAILY_LOG, exist_ok=True)

    fetcher = DataFetcher()
    store = PatternStore()
    dashboard = LiveDashboard()
    state = AnalysisState()

    if not fetcher.try_saved_token():
        if not fetcher.authenticate():
            print("\n❌ Auth failed. Exiting.")
            sys.exit(1)

    if not fetcher.load_instruments():
        print("\n❌ No instruments. Exiting.")
        sys.exit(1)

    nifty_price = fetcher.get_nifty_price()
    if nifty_price is None:
        print("\n⏳ Waiting for market...")
        while nifty_price is None:
            if fetcher.is_market_open():
                nifty_price = fetcher.get_nifty_price()
            if nifty_price is None:
                time.sleep(30)

    print(f"\n   Nifty spot: {nifty_price}")
    fetcher.build_chain(nifty_price)
    fetcher.fetch_nifty_history(days=60)

    nifty_closes = [c["close"] for c in fetcher.nifty_candles] if fetcher.nifty_candles else []
    prev_enriched = None
    candle_count = 0

    print("\n📊 Opening Excel...")
    dashboard.setup()

    stats = store.get_stats()
    total = stats["total_patterns"]
    mat = "BABY" if total < 100 else ("LEARNING" if total < 500 else
                                      ("READY" if total < 1000 else "MATURE"))

    print(f"\n{'=' * 60}")
    print(f"   SYSTEM v2 RUNNING — {mat}")
    print(f"   Patterns: {total} | Math: All 9 concepts active")
    print(f"   Kalman ✅ | PCA ✅ | Transfer Entropy ✅")
    print(f"   Implied Distribution ✅ | Hurst-Edge ✅ | Exit Signals ✅")
    print(f"   Press Ctrl+C to stop")
    print(f"{'=' * 60}")

    try:
        while True:
            if not fetcher.is_market_open():
                now = datetime.now()
                if now.hour >= 15 and now.minute > 35:
                    print("\n\n🔔 Market closed. End-of-day...")
                    eod = store.end_of_day_update()
                    print(f"   Patterns: {eod['total_patterns']} | "
                          f"Added: {eod['added_today']} | Pruned: {eod['pruned']}")

                    daily_file = os.path.join(config.DAILY_LOG,
                                              f"day_{now.strftime('%Y%m%d')}.json")
                    with open(daily_file, "w") as f:
                        json.dump({"date": now.strftime("%Y-%m-%d"),
                                   "candles": candle_count}, f)
                    dashboard.close()
                    print("\n   Done. See you tomorrow.")
                    break
                else:
                    print(f"   Waiting... ({now.strftime('%H:%M:%S')})")
                    time.sleep(30)
                    continue

            wait_time = fetcher.seconds_to_next_candle()
            candle_num = fetcher.get_candle_number()

            if candle_count > 0:
                print(f"\n⏳ Next candle in {wait_time:.0f}s...")
                time.sleep(wait_time)

            candle_count += 1

            print(f"\n📊 Candle #{candle_num} — Fetching...")
            snapshot = fetcher.fetch_live_snapshot()
            if snapshot is None:
                print("   ⚠️ Failed. Skipping.")
                continue

            nifty_price = snapshot["nifty_price"]
            nifty_closes.append(nifty_price)

            new_atm = round(nifty_price / config.STRIKE_GAP) * config.STRIKE_GAP
            if new_atm != fetcher.atm_strike:
                print(f"   ATM: {fetcher.atm_strike} → {new_atm}")
                fetcher.build_chain(nifty_price)
                snapshot = fetcher.fetch_live_snapshot()
                if snapshot is None:
                    continue

            dte = fetcher.get_days_to_expiry()

            # enrich with Kalman filter
            enriched = enrich_chain(snapshot, dte, state)

            # full probability calculation
            prob_result = calculate_probability(
                nifty_closes, enriched, fetcher.atm_strike,
                fetcher.option_snapshots, prev_enriched,
                store, state, dte
            )

            # Hurst-adjusted edge
            edges = calculate_edge(
                enriched, nifty_price, prob_result["prob_up"],
                dte, prob_result.get("hurst", 0.5)
            )

            # constrained strike optimization
            best_picks = optimize_strike(
                enriched, edges, prob_result["prob_up"],
                fetcher.atm_strike, dte
            )

            # console output
            print_signal(prob_result, nifty_price, fetcher.atm_strike, best_picks)

            # excel
            success = dashboard.update(
                nifty_price, fetcher.atm_strike, prob_result,
                enriched, edges, candle_num, store.get_stats(),
                dte, str(fetcher.current_expiry), best_picks
            )
            if success:
                print(f"   ✅ Excel updated")

            # learn
            store.store_candle_data(nifty_closes, prob_result, snapshot)
            store.fill_results(nifty_closes)

            prev_enriched = enriched

    except KeyboardInterrupt:
        print("\n\n🛑 Stopped.")
        store.save()
        dashboard.close()
        print("   Saved. Done!")


if __name__ == "__main__":
    main()