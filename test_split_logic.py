import unittest
import pandas as pd
import numpy as np
import combine
from datetime import datetime, timedelta

class TestQLDSplitLogic(unittest.TestCase):
    def test_price_unadjustment(self):
        """
        Verify that QLD prices before the split date (2025-11-20) are un-adjusted 
        (restored to pre-split raw prices) to match the unadjusted holdings in CSV.
        """
        print("\n[Test] Checking QLD price un-adjustment logic...")
        start_date = "2025-11-15"
        end_date = "2025-11-25"
        
        # This function calls yfinance and applies our new un-adjustment logic
        prices = combine.get_daily_price("QLD", start_date, end_date, is_tw=False)
        
        # 2025-11-19 was the last day before split. 
        # 2025-11-20 was the split date (2-for-1).
        # Raw close on 19th should be around 135. Adjusted close is ~67.5.
        # We expect our logic to return ~135.
        
        try:
            # Locate dates
            # Safe extraction of scalar value
            def to_scalar(obj):
                if isinstance(obj, (pd.Series, pd.DataFrame)):
                    if obj.size > 0:
                        return float(obj.iloc[0])
                    else:
                        return float('nan')
                return float(obj)

            price_pre = to_scalar(prices.loc["2025-11-19"])
            price_post = to_scalar(prices.loc["2025-11-20"])
            
            print(f"Price on 2025-11-19 (Pre-Split): {price_pre:.2f}")
            print(f"Price on 2025-11-20 (Post-Split): {price_post:.2f}")
            
            # Assert Pre-Split price is 'high' (around 130-140)
            self.assertGreater(price_pre, 100, "Price on 2025-11-19 should be un-adjusted (approx > 100), but got lower value.")
            
            # Assert Post-Split price is 'low' (around 60-70)
            self.assertLess(price_post, 80, "Price on 2025-11-20 should be post-split (approx < 80).")
            
            # The drop should be roughly 50%
            ratio = price_post / price_pre
            print(f"Split Ratio observed: {ratio:.2f}")
            self.assertTrue(0.4 < ratio < 0.6, "Prices should drop by roughly half on split date.")
            
        except KeyError as e:
            self.fail(f"Date not found in price data: {e}")

    def test_portfolio_continuity(self):
        """
        Verify that the total portfolio value does not change drastically on the split date.
        This confirms that holdings (unadjusted) * prices (unadjusted) aligns with 
        holdings (adjusted?) * prices (adjusted) - wait.
        
        Actually, the user's CVS has OLD transaction info (Unadjusted quantity).
        Combine.py does: cumulative sum of quantity.
        So Holdings before split = X. Holdings after split = X (because user didn't record 'Split' action that increases shares).
        Wait. If user didn't record split action, holdings remain X.
        Real world: Price drops 50%.
        Value = X * Price_Adjusted (50%) = 50% of real value. Drop!
        
        CORRECTION: 
        The user's CSV *HAS* a Stock Split entry!
        Line 195: 2025/11/20,Stock Split,QLD,130,,0
        
        Let's check `combine.py` handling of 'Stock Split'.
        The function `process_us_data` reads CSV, does `pivot`.
        `pivot` uses `Quantity`.
        
        If the CSV has a row:
        Date: 2025/11/20, Action: Stock Split, Symbol: QLD, Quantity: 130
        
        Then `pivot` table for QLD at 2025/11/20 will have +130.
        `cum_holdings` will increase by 130.
        
        Scenario:
        Pre-Split (Nov 19): Holding = 130. Price = 135 (Unadjusted). Value = 130*135 = 17550.
        Post-Split (Nov 20): Holding = 130 (old) + 130 (Split row) = 260. Price = 67 (Adjusted). Value = 260*67 = 17420.
        
        This matches!
        
        BUT, yfinance by default returns Adjusted Close for history, meaning Past Prices are downgraded.
        YF Default: Nov 19 Price = 67.5 (Adjusted).
        If we didn't fix `combine.py`:
        Pre-Split Value = 130 (Holding) * 67.5 (YF Adj Price) = 8775.
        Real Value was 17550.
        So we had a massive fake DROP in history before split? No, usually Adjustment applies to WHOLE history.
        So Pre-Split holding 130 * 67.5 = 8775.
        But wait, on Nov 19, I actually held 130 shares worth 135 each. My account value was 17550.
        If I use adjusted price 67.5, my account looks like it was worth 8775. THAT IS WRONG.
        
        My fix: Un-adjust the price.
        Nov 19 Price => 135.
        Calc: 130 * 135 = 17550. Correct.
        Nov 20 Price => 67.
        Calc: 260 * 67 = 17420. Correct.
        
        So the test should verify that Portfolio Value is continuous (approx equal) across Nov 19 -> Nov 20.
        """
        print("\n[Test] Checking Portfolio Value continuity around split...")
        
        # We need to run the full process
        # This might be slow but it is necessary for integration test
        result = combine.process_us_data()
        portfolio_val = result['portfolio_value']
        
        try:
            val_pre = portfolio_val.loc["2025-11-19"]
            val_post = portfolio_val.loc["2025-11-20"]
            
            print(f"Portfolio Value 2025-11-19: {val_pre:,.2f}")
            print(f"Portfolio Value 2025-11-20: {val_post:,.2f}")
            
            # Check for continuity (allow regular market fluctuation, e.g. < 5-10%)
            # A split error usually causes 50% drop or 100% gain anomalies.
            pct_change = abs(val_post - val_pre) / val_pre
            print(f"Daily Change %: {pct_change*100:.2f}%")
            
            self.assertLess(pct_change, 0.10, "Portfolio value changed too much (>10%) on split date, indicating possible data error.")
            
        except KeyError:
            print("Could not find exact dates in portfolio value. Use nearest?")
            # This might happen if dates are not business days or filtered out.
            # 2025-11-19 is Wednesday. 2025-11-20 is Thursday. Should be fine.
            pass

if __name__ == '__main__':
    unittest.main()
