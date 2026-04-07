import time
import xlwings as xw
from datetime import datetime
import config


class LiveDashboard:
    def __init__(self):
        self.wb = None
        self.ws = None
        self.app = None
        self.is_setup = False

    def setup(self):
        try:
            # kill zombie Excel
            import subprocess
            try:
                subprocess.run(['taskkill', '/f', '/im', 'EXCEL.EXE'],
                               capture_output=True, timeout=5)
                time.sleep(1)
            except:
                pass

            # delete old file
            import os
            for f in [config.EXCEL_OUTPUT, config.EXCEL_OUTPUT.replace(".xlsx", "_live.xlsx")]:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except:
                    pass

            # COM init
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except:
                pass

            self.app = xw.App(visible=True)
            self.app.display_alerts = False
            time.sleep(1)

            self.wb = self.app.books.add()
            self.ws = self.wb.sheets[0]
            self.ws.name = "Dashboard"

            for c, w in {"A": 14, "B": 10, "C": 10, "D": 10, "E": 10,
                         "F": 10, "G": 10, "H": 10, "I": 14, "J": 14}.items():
                self.ws.range(f"{c}:{c}").column_width = w

            try:
                self.wb.save(config.EXCEL_OUTPUT)
            except:
                alt = config.EXCEL_OUTPUT.replace(".xlsx", "_live.xlsx")
                self.wb.save(alt)
                print(f"   ⚠️ Saved as {alt}")

            self.is_setup = True
            print(f"   ✅ Excel opened")
            return True

        except Exception as e:
            print(f"   ❌ Excel setup failed: {e}")
            print(f"   Try: close all Excel, delete dashboard.xlsx, run again")
            return False

    def _screen(self, val):
        for attempt in range(3):
            try:
                self.app.screen_updating = val
                return
            except:
                time.sleep(0.5 * (attempt + 1))

    def _w(self, row, col, val, bold=False, size=10, bg=None, fg=None):
        for attempt in range(2):
            try:
                c = self.ws.range((row, col))
                c.value = val
                c.font.size = size
                c.font.bold = bold
                if bg:
                    c.color = bg
                if fg:
                    c.font.color = fg
                return
            except:
                time.sleep(0.3)

    def _row(self, row, vals, **kwargs):
        for i, v in enumerate(vals, 1):
            self._w(row, i, v, **kwargs)

    def _clear(self, row, cols=10):
        for c in range(1, cols + 1):
            try:
                cell = self.ws.range((row, c))
                cell.value = None
                cell.color = None
                cell.font.bold = False
                cell.font.color = (0, 0, 0)
            except:
                pass

    def update(self, nifty_price, atm_strike, prob_result, enriched_chain,
               edges, candle_number, store_stats, days_to_expiry, expiry_date,
               best_picks=None):

        if not self.is_setup:
            if not self.setup():
                return False

        try:
            self._screen(False)

            pu = prob_result["prob_up"]
            pd = prob_result["prob_down"]
            strength = prob_result["strength"]
            regime = prob_result["regime"]
            conf = prob_result["confidence"]
            hurst = prob_result.get("hurst", 0.5)
            voters = prob_result.get("voters", {})
            regime_probs = prob_result.get("regime_probs", {})
            exits = prob_result.get("exit_signals", [])
            pca = prob_result.get("pca", {})

            G = (198, 239, 206)
            R = (255, 199, 206)
            Y = (255, 235, 156)
            B = (189, 215, 238)
            D = (47, 84, 150)
            H = (217, 226, 243)
            GR = (100, 100, 100)

            if pu > 55:
                direction = "BULLISH UP"
                pc = G
            elif pu < 45:
                direction = "BEARISH DOWN"
                pc = R
            else:
                direction = "NEUTRAL"
                pc = Y

            time_str = datetime.now().strftime("%H:%M:%S")
            row = 1

            # title
            self._row(row, ["NIFTY OPTIONS v2", "", "", "", "", "", "", "", "", ""],
                      bold=True, size=13, bg=D, fg=(255, 255, 255))
            row += 1

            # info
            self._row(row, [
                f"NIFTY: {nifty_price:.0f}", f"ATM: {atm_strike}",
                f"Time: {time_str}", f"Candle: #{candle_number}",
                f"Expiry: {expiry_date}", f"DTE: {days_to_expiry:.3f}yr",
                f"Hurst: {hurst:.2f}", "", "", ""
            ], bold=True, size=10, bg=B)
            row += 1

            # probability
            self._row(row, [
                direction, f"UP: {pu}%", f"DOWN: {pd}%",
                f"Strength: {strength}", f"Regime: {regime}",
                f"Conf: {conf}", "", "", "", ""
            ], bold=True, size=12, bg=pc)
            row += 1

            # status
            lib = store_stats.get("total_patterns", 0)
            mem = store_stats.get("memory_mb", 0)
            mat = "BABY" if lib < 100 else ("LEARNING" if lib < 500 else
                                            ("READY" if lib < 1000 else "MATURE"))
            pca_exp = pca.get("explained_ratio")
            pca_str = f"PCA: {pca_exp:.0%}" if pca_exp else "PCA: --"

            self._row(row, [
                f"Maturity: {mat}", f"Patterns: {lib}", f"Mem: {mem}MB",
                f"Today: {store_stats.get('today_records', 0)}",
                pca_str, "", "", "", "", ""
            ], size=9, fg=GR)
            row += 1

            # regime probs
            if regime_probs:
                self._row(row, [
                    f"TC:{regime_probs.get('trending_calm', 0):.0%}",
                    f"TV:{regime_probs.get('trending_volatile', 0):.0%}",
                    f"RC:{regime_probs.get('ranging_calm', 0):.0%}",
                    f"RV:{regime_probs.get('ranging_volatile', 0):.0%}",
                    "", "", "", "", "", ""
                ], size=9, fg=GR)
            row += 1

            # voters
            vv = [f"{k}:{v['vote']:.0f}%" for k, v in voters.items()]
            while len(vv) < 10:
                vv.append("")
            self._row(row, vv, size=9, fg=GR)
            row += 1

            # exit alerts
            if exits:
                for ex in exits[:3]:
                    self._row(row, [
                        f"EXIT: {ex['type']}", ex['message'],
                        "", "", "", "", "", "", "", ""
                    ], bold=True, size=10, bg=R)
                    row += 1

            # best picks
            if best_picks:
                real_picks = [p for p in best_picks if p.get("signal", "")]

                if real_picks:
                    self._row(row, [
                        "TOP PICKS:", "", "", "", "", "", "", "", "", ""
                    ], bold=True, size=10, bg=G)
                    row += 1
                    for pick in real_picks[:3]:
                        signal = pick.get("signal", "")
                        sp = pick.get("strike_prob", 0)

                        if "STRONG" in signal:
                            pick_bg = (198, 239, 206)
                        elif "BUY" in signal:
                            pick_bg = (210, 240, 210)
                        elif "watch" in signal:
                            pick_bg = (255, 235, 156)
                        else:
                            pick_bg = (230, 230, 230)

                        self._row(row, [
                            signal,
                            f"{pick['strike']} {pick['type']}",
                            f"LTP: {pick['ltp']}",
                            f"Edge: {pick['edge']:+.1f}",
                            f"Prob: {sp:.0f}%",
                            f"Delta: {pick['delta']:.2f}",
                            f"Score: {pick['score']}",
                            f"Vol: {pick['volume']}",
                            f"Spd: {pick['spread']:.1f}",
                            ""
                        ], size=10, bg=pick_bg)
                        row += 1
                else:
                    self._row(row, [
                        "NO PICKS — prob too low or no edge",
                        "", "", "", "", "", "", "", "", ""
                    ], size=10, bg=Y)
                    row += 1
            else:
                self._row(row, [
                    "NO PICKS — prob too low or no edge",
                    "", "", "", "", "", "", "", "", ""
                ], size=10, bg=Y)
                row += 1

            row += 1

            # table headers
            headers = ["#", "Strike", "CE LTP", "CE Prob%",
                       "CE Edge", "PE LTP", "PE Prob%",
                       "PE Edge", "Pick", "Note"]
            self._row(row, headers, bold=True, size=10, bg=H)
            row += 1

            # chain
            sorted_strikes = sorted(enriched_chain.keys(), reverse=True)

            for idx, strike in enumerate(sorted_strikes, 1):
                types = enriched_chain[strike]
                strike_edges = edges.get(strike, {})
                is_atm = abs(strike - atm_strike) <= config.STRIKE_GAP // 2

                ce = types.get("CE", {})
                pe = types.get("PE", {})
                ce_ltp = ce.get("ltp", 0)
                pe_ltp = pe.get("ltp", 0)
                ce_delta = ce.get("delta")
                pe_delta = pe.get("delta")
                ce_edge = strike_edges.get("CE")
                pe_edge = strike_edges.get("PE")
                ce_vol = ce.get("volume", 0)
                pe_vol = pe.get("volume", 0)

                # per-strike probability
                if ce_ltp > 1 and ce_vol > 50 and ce_delta and abs(ce_delta) > 0.03:
                    ce_sp = min(95, max(5, abs(ce_delta) * (pu / 50) * 100))
                    ce_prob_str = f"{ce_sp:.0f}%"
                else:
                    ce_prob_str = "--"

                if pe_ltp > 1 and pe_vol > 50 and pe_delta and abs(pe_delta) > 0.03:
                    pe_sp = min(95, max(5, abs(pe_delta) * (pd / 50) * 100))
                    pe_prob_str = f"{pe_sp:.0f}%"
                else:
                    pe_prob_str = "--"

                ce_edge_str = f"{ce_edge:+.1f}" if ce_edge is not None and ce_vol > 50 else "--"
                pe_edge_str = f"{pe_edge:+.1f}" if pe_edge is not None and pe_vol > 50 else "--"

                # pick from optimizer
                pick = ""
                pick_bg = None
                if best_picks:
                    for bp in best_picks[:3]:
                        if bp["strike"] == strike and bp.get("signal", ""):
                            pick = bp["signal"]
                            if "STRONG" in pick:
                                pick_bg = G
                            elif "BUY" in pick:
                                pick_bg = (210, 240, 210)
                            elif "watch" in pick:
                                pick_bg = Y

                # note
                note = ""
                ce_kalman = ce.get("kalman_price", ce_ltp)
                if ce_ltp > 0 and abs(ce_kalman - ce_ltp) / max(ce_ltp, 1) > 0.02:
                    note = "Noisy"

                values = [idx, strike, ce_ltp, ce_prob_str, ce_edge_str,
                          pe_ltp, pe_prob_str, pe_edge_str, pick, note]

                row_bg = B if is_atm else None
                self._row(row, values, bold=is_atm, size=10, bg=row_bg)

                if ce_edge is not None and ce_vol > 50:
                    if ce_edge > 2:
                        self._w(row, 5, ce_edge_str, bg=G)
                    elif ce_edge < -2:
                        self._w(row, 5, ce_edge_str, bg=R)

                if pe_edge is not None and pe_vol > 50:
                    if pe_edge > 2:
                        self._w(row, 8, pe_edge_str, bg=G)
                    elif pe_edge < -2:
                        self._w(row, 8, pe_edge_str, bg=R)

                if pick and pick_bg:
                    self._w(row, 9, pick, bold="STRONG" in pick, bg=pick_bg)

                row += 1

            # clear old rows
            for cr in range(row, row + 10):
                self._clear(cr)

            # legend
            row += 1
            self._row(row, ["Prob% = per-strike (delta × direction)", "", "", "",
                            "", "", "", "", "", ""], size=9, fg=GR)
            row += 1
            self._row(row, ["Edge = Hurst-adjusted BS reprice (points ₹)", "", "", "",
                            "", "", "", "", "", ""], size=9, fg=GR)
            row += 1
            self._row(row, [f"Updates every {config.CANDLE_INTERVAL} min | DO NOT close",
                            "", "", "", "", "", "", "", "", ""], size=9, fg=GR)

            self._screen(True)

            try:
                self.wb.save()
            except:
                pass

            return True

        except Exception as e:
            print(f"   ❌ Excel error: {e}")
            self._screen(True)
            try:
                self.is_setup = False
                self.setup()
            except:
                pass
            return False

    def close(self):
        try:
            if self.wb:
                self.wb.save()
            if self.app:
                self._screen(True)
        except:
            pass