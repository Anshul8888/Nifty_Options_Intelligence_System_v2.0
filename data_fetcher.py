import os
import json
import time
import webbrowser
from datetime import datetime, timedelta

import pandas as pd
from kiteconnect import KiteConnect

import config


class DataFetcher:
    def __init__(self):
        self.kite = KiteConnect(api_key=config.API_KEY)
        self.access_token = None
        self.instruments_df = None
        self.current_expiry = None
        self.chain_tokens = {}          # {strike: {"CE": token, "PE": token}}
        self.nifty_candles = []          # historical 3-min candles
        self.option_snapshots = []       # list of snapshots over the day
        self.atm_strike = None

    # ----------------------------------------------------------
    # AUTHENTICATION
    # ----------------------------------------------------------
    def authenticate(self):
        """Daily login flow — opens browser, user pastes request token."""
        login_url = self.kite.login_url()
        print("\n" + "=" * 60)
        print("STEP 1: Login to Zerodha")
        print("=" * 60)
        print(f"\nOpening browser. If it doesn't open, go to:\n{login_url}\n")
        webbrowser.open(login_url)

        print("After logging in, you'll be redirected.")
        print("Look at the URL bar. Find 'request_token=XXXXXX'")
        request_token = input("\nPaste the request_token here: ").strip()

        try:
            data = self.kite.generate_session(request_token, api_secret=config.API_SECRET)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            print(f"\n✅ Authenticated successfully!")
            print(f"   Access token valid for today.")

            # save token for potential restart
            with open(".access_token", "w") as f:
                f.write(self.access_token)

            return True
        except Exception as e:
            print(f"\n❌ Authentication failed: {e}")
            return False

    def try_saved_token(self):
        """Try to use a saved access token from today."""
        if os.path.exists(".access_token"):
            with open(".access_token", "r") as f:
                token = f.read().strip()
            self.kite.set_access_token(token)
            try:
                self.kite.profile()
                self.access_token = token
                print("✅ Using saved access token from earlier today.")
                return True
            except:
                pass
        return False

    # ----------------------------------------------------------
    # INSTRUMENT MANAGEMENT
    # ----------------------------------------------------------
    def load_instruments(self):
        """Download NFO instrument master and find current expiry."""
        print("\n📋 Downloading instrument master...")
        instruments = self.kite.instruments(config.OPTIONS_EXCHANGE)
        self.instruments_df = pd.DataFrame(instruments)

        # filter for NIFTY options only
        nifty_opts = self.instruments_df[
            (self.instruments_df["name"] == "NIFTY") &
            (self.instruments_df["instrument_type"].isin(["CE", "PE"]))
            ].copy()

        # find nearest expiry (today or future)
        today = datetime.now().date()
        future_expiries = nifty_opts[nifty_opts["expiry"] >= today]["expiry"].unique()
        future_expiries.sort()

        if len(future_expiries) == 0:
            print("❌ No active expiries found!")
            return False

        self.current_expiry = future_expiries[0]
        print(f"   Current expiry: {self.current_expiry}")
        print(f"   Days to expiry: {(self.current_expiry - today).days}")

        return True

    def get_nifty_price(self):
        """Fetch current Nifty spot price."""
        try:
            quote = self.kite.quote([f"{config.NIFTY_EXCHANGE}:{config.NIFTY_SYMBOL}"])
            key = f"{config.NIFTY_EXCHANGE}:{config.NIFTY_SYMBOL}"
            return quote[key]["last_price"]
        except Exception as e:
            print(f"❌ Error fetching Nifty price: {e}")
            return None

    def build_chain(self, nifty_price):
        """Build options chain: 20 strikes above and below ATM."""
        self.atm_strike = round(nifty_price / config.STRIKE_GAP) * config.STRIKE_GAP

        strikes = []
        for i in range(-config.STRIKES_BELOW_ATM, config.STRIKES_ABOVE_ATM + 1):
            strikes.append(int(self.atm_strike + i * config.STRIKE_GAP))

        # find instrument tokens for each strike
        nifty_opts = self.instruments_df[
            (self.instruments_df["name"] == "NIFTY") &
            (self.instruments_df["expiry"] == self.current_expiry) &
            (self.instruments_df["instrument_type"].isin(["CE", "PE"]))
            ]

        self.chain_tokens = {}
        found_count = 0

        for strike in strikes:
            self.chain_tokens[strike] = {"CE": None, "PE": None}
            for opt_type in ["CE", "PE"]:
                row = nifty_opts[
                    (nifty_opts["strike"] == strike) &
                    (nifty_opts["instrument_type"] == opt_type)
                    ]
                if len(row) > 0:
                    self.chain_tokens[strike][opt_type] = int(row.iloc[0]["instrument_token"])
                    found_count += 1

        print(f"\n📊 Options chain built:")
        print(f"   ATM: {self.atm_strike}")
        print(f"   Range: {strikes[0]} to {strikes[-1]}")
        print(f"   Strikes: {len(strikes)}")
        print(f"   Contracts found: {found_count}")

        return True

    # ----------------------------------------------------------
    # DATA FETCHING
    # ----------------------------------------------------------
    def fetch_live_snapshot(self):
        """Fetch live data for Nifty + all options in chain."""
        # build list of all instrument tokens
        tokens = [config.NIFTY_TOKEN]
        token_to_strike = {}

        for strike, types in self.chain_tokens.items():
            for opt_type in ["CE", "PE"]:
                token = types[opt_type]
                if token:
                    tokens.append(token)
                    token_to_strike[token] = (strike, opt_type)

        try:
            quotes = self.kite.quote(tokens)
        except Exception as e:
            print(f"❌ Error fetching quotes: {e}")
            return None

        # parse nifty data
        nifty_key = str(config.NIFTY_TOKEN)
        # kite.quote with token returns string keys
        nifty_data = None
        for key, val in quotes.items():
            if val.get("instrument_token") == config.NIFTY_TOKEN:
                nifty_data = val
                break

        if nifty_data is None:
            print("❌ Nifty data not found in quotes")
            return None

        nifty_price = nifty_data["last_price"]

        # parse options data
        options_data = {}
        for key, val in quotes.items():
            token = val.get("instrument_token")
            if token in token_to_strike:
                strike, opt_type = token_to_strike[token]
                if strike not in options_data:
                    options_data[strike] = {}
                options_data[strike][opt_type] = {
                    "ltp": val.get("last_price", 0),
                    "oi": val.get("oi", 0),
                    "volume": val.get("volume", 0),
                    "buy_qty": val.get("buy_quantity", 0),
                    "sell_qty": val.get("sell_quantity", 0),
                    "bid": val.get("depth", {}).get("buy", [{}])[0].get("price", 0),
                    "ask": val.get("depth", {}).get("sell", [{}])[0].get("price", 0),
                    "last_qty": val.get("last_quantity", 0),
                    "avg_price": val.get("average_price", 0),
                    "ohlc": val.get("ohlc", {}),
                }

        snapshot = {
            "timestamp": datetime.now(),
            "nifty_price": nifty_price,
            "nifty_ohlc": nifty_data.get("ohlc", {}),
            "atm": self.atm_strike,
            "options": options_data
        }

        self.option_snapshots.append(snapshot)
        return snapshot

    def fetch_nifty_history(self, days=60):
        """Fetch historical 3-minute candles for Nifty."""
        print(f"\n📈 Fetching {days} days of Nifty 3-min history...")
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        try:
            # Kite limits historical data to 60 days for minute candles
            # fetch in chunks if needed
            all_candles = []
            chunk_start = from_date

            while chunk_start < to_date:
                chunk_end = min(chunk_start + timedelta(days=55), to_date)
                candles = self.kite.historical_data(
                    instrument_token=config.NIFTY_TOKEN,
                    from_date=chunk_start.strftime("%Y-%m-%d"),
                    to_date=chunk_end.strftime("%Y-%m-%d"),
                    interval="3minute"
                )
                all_candles.extend(candles)
                chunk_start = chunk_end + timedelta(days=1)
                time.sleep(0.5)     # rate limit courtesy

            self.nifty_candles = all_candles
            print(f"   Loaded {len(all_candles)} candles")
            print(f"   From: {all_candles[0]['date'] if all_candles else 'N/A'}")
            print(f"   To: {all_candles[-1]['date'] if all_candles else 'N/A'}")
            return True

        except Exception as e:
            print(f"⚠️ Could not fetch full history: {e}")
            print("   System will work with limited history.")
            return False

    def get_days_to_expiry(self):
        """Calculate days to expiry as fraction."""
        if self.current_expiry is None:
            return 1 / 365
        today = datetime.now()
        expiry_dt = datetime.combine(self.current_expiry, datetime.min.time().replace(hour=15, minute=30))
        diff = (expiry_dt - today).total_seconds()
        return max(diff / (365 * 24 * 3600), 1 / (365 * 24))  # minimum 1 hour

    def is_market_open(self):
        """Check if market is currently open."""
        now = datetime.now()
        if now.weekday() >= 5:  # Saturday, Sunday
            return False
        market_open = now.replace(hour=9, minute=15, second=0)
        market_close = now.replace(hour=15, minute=30, second=0)
        return market_open <= now <= market_close

    def seconds_to_next_candle(self):
        """Calculate seconds until next 3-minute candle boundary."""
        now = datetime.now()
        market_open = now.replace(hour=9, minute=15, second=0)
        elapsed = (now - market_open).total_seconds()
        interval = config.CANDLE_INTERVAL * 60
        current_candle = int(elapsed / interval)
        next_candle_time = market_open + timedelta(seconds=(current_candle + 1) * interval)
        wait = (next_candle_time - now).total_seconds() + 2  # 2 sec buffer
        return max(wait, 1)

    def get_candle_number(self):
        """Which candle of the day are we on."""
        now = datetime.now()
        market_open = now.replace(hour=9, minute=15, second=0)
        elapsed = (now - market_open).total_seconds()
        return int(elapsed / (config.CANDLE_INTERVAL * 60)) + 1