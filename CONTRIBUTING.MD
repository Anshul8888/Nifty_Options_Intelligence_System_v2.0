🤝 Contributing
Found a bug? Have an improvement?

Current known limitations:

Doesn't handle splits/bonus/corporate actions (rare in Nifty options)
Transfer entropy needs 36+ minutes to activate (acceptable for intraday)
Excel COM errors on some Windows versions (retry logic helps but not perfect)
Pattern matching correlation threshold is debatable (current: 0.55)
Potential enhancements:

Add support for Bank Nifty
Add multi-timeframe analysis (5-min + 15-min)
Add position tracker (track P&L of executed trades)
Add backtesting module
Add Telegram alerts for STRONG signals
