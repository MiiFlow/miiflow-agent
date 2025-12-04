"""Yahoo Finance tools for fetching stock market data.

These tools use the yfinance library to fetch real-time and historical
stock data. No API key is required.

Install: pip install yfinance
"""

from typing import Optional
from miiflow_llm.core.tools import tool


@tool("get_stock_quote", "Get real-time stock quote and key metrics for a symbol")
def get_stock_quote(symbol: str) -> str:
    """Fetch current stock price and basic metrics.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')

    Returns:
        Formatted string with current price, change, and key metrics
    """
    try:
        import yfinance as yf
    except ImportError:
        return "Error: yfinance not installed. Run: pip install yfinance"

    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info

        if not info or "regularMarketPrice" not in info:
            # Try fast_info as fallback
            fast = ticker.fast_info
            if hasattr(fast, "last_price") and fast.last_price:
                return f"""Stock Quote for {symbol.upper()}:
- Current Price: ${fast.last_price:.2f}
- Market Cap: ${fast.market_cap:,.0f} (if available)
Note: Limited data available. Try a different symbol or check market hours."""

            return f"Unable to fetch data for symbol '{symbol}'. Please verify the ticker symbol is correct."

        # Extract key metrics
        current_price = info.get("regularMarketPrice", info.get("currentPrice", "N/A"))
        previous_close = info.get("regularMarketPreviousClose", info.get("previousClose", "N/A"))

        # Calculate change
        if isinstance(current_price, (int, float)) and isinstance(previous_close, (int, float)):
            change = current_price - previous_close
            change_pct = (change / previous_close) * 100
            change_str = f"${change:+.2f} ({change_pct:+.2f}%)"
        else:
            change_str = "N/A"

        # Format market cap
        market_cap = info.get("marketCap", "N/A")
        if isinstance(market_cap, (int, float)):
            if market_cap >= 1e12:
                market_cap_str = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                market_cap_str = f"${market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                market_cap_str = f"${market_cap/1e6:.2f}M"
            else:
                market_cap_str = f"${market_cap:,.0f}"
        else:
            market_cap_str = "N/A"

        # Format volume
        volume = info.get("regularMarketVolume", info.get("volume", "N/A"))
        if isinstance(volume, (int, float)):
            if volume >= 1e6:
                volume_str = f"{volume/1e6:.2f}M"
            else:
                volume_str = f"{volume:,.0f}"
        else:
            volume_str = "N/A"

        # Helper function to format numeric values
        def fmt_price(val, prefix="$"):
            if isinstance(val, (int, float)):
                return f"{prefix}{val:.2f}"
            return "N/A"

        def fmt_num(val):
            if isinstance(val, (int, float)):
                return f"{val:.2f}"
            return "N/A"

        # Format day range
        day_low = info.get('dayLow')
        day_high = info.get('dayHigh')
        day_range = f"{fmt_price(day_low)} - {fmt_price(day_high)}"

        # Format 52-week range
        week_low = info.get('fiftyTwoWeekLow')
        week_high = info.get('fiftyTwoWeekHigh')
        week_range = f"{fmt_price(week_low)} - {fmt_price(week_high)}"

        # Format P/E and EPS
        pe_ratio = info.get('trailingPE')
        pe_str = fmt_num(pe_ratio)
        eps = info.get('trailingEps')
        eps_str = fmt_price(eps)

        # Format dividend yield
        # yfinance returns dividendYield as a percentage value (e.g., 0.37 means 0.37%)
        div_yield = info.get('dividendYield')
        if isinstance(div_yield, (int, float)) and div_yield > 0:
            div_str = f"{div_yield:.2f}%"
        else:
            div_str = "N/A"

        result = f"""Stock Quote for {info.get('shortName', symbol.upper())} ({symbol.upper()}):
- Current Price: {fmt_price(current_price)}
- Change: {change_str}
- Previous Close: {fmt_price(previous_close)}
- Day Range: {day_range}
- 52 Week Range: {week_range}
- Volume: {volume_str}
- Market Cap: {market_cap_str}
- P/E Ratio: {pe_str}
- EPS: {eps_str}
- Dividend Yield: {div_str}"""

        return result

    except Exception as e:
        return f"Error fetching quote for {symbol}: {str(e)}"


@tool("get_stock_history", "Get historical stock price data")
def get_stock_history(symbol: str, period: str = "1mo") -> str:
    """Fetch historical price data for a stock.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        period: Time period - valid options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, ytd, max

    Returns:
        Summary of historical price data including performance metrics
    """
    try:
        import yfinance as yf
    except ImportError:
        return "Error: yfinance not installed. Run: pip install yfinance"

    valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"]
    if period not in valid_periods:
        return f"Invalid period '{period}'. Valid options: {', '.join(valid_periods)}"

    try:
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period=period)

        if hist.empty:
            return f"No historical data available for {symbol} over period {period}"

        # Calculate metrics
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        high_price = hist['High'].max()
        low_price = hist['Low'].min()
        avg_volume = hist['Volume'].mean()

        change = end_price - start_price
        change_pct = (change / start_price) * 100

        # Format volume
        if avg_volume >= 1e6:
            avg_volume_str = f"{avg_volume/1e6:.2f}M"
        else:
            avg_volume_str = f"{avg_volume:,.0f}"

        # Get date range
        start_date = hist.index[0].strftime('%Y-%m-%d')
        end_date = hist.index[-1].strftime('%Y-%m-%d')

        result = f"""Historical Data for {symbol.upper()} ({period}):
Period: {start_date} to {end_date} ({len(hist)} trading days)

Price Summary:
- Starting Price: ${start_price:.2f}
- Ending Price: ${end_price:.2f}
- Change: ${change:+.2f} ({change_pct:+.2f}%)
- Period High: ${high_price:.2f}
- Period Low: ${low_price:.2f}
- Average Daily Volume: {avg_volume_str}

Performance: {"Positive" if change >= 0 else "Negative"} trend over this period."""

        return result

    except Exception as e:
        return f"Error fetching history for {symbol}: {str(e)}"


@tool("get_company_info", "Get company profile and business information")
def get_company_info(symbol: str) -> str:
    """Fetch company profile information.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')

    Returns:
        Company profile including sector, industry, and description
    """
    try:
        import yfinance as yf
    except ImportError:
        return "Error: yfinance not installed. Run: pip install yfinance"

    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info

        if not info or "shortName" not in info:
            return f"Unable to fetch company info for '{symbol}'. Please verify the ticker symbol."

        # Format employee count
        employees = info.get('fullTimeEmployees', 'N/A')
        if isinstance(employees, int):
            if employees >= 1000:
                employees_str = f"{employees:,}"
            else:
                employees_str = str(employees)
        else:
            employees_str = "N/A"

        result = f"""Company Profile: {info.get('shortName', symbol.upper())} ({symbol.upper()})

Basic Info:
- Full Name: {info.get('longName', 'N/A')}
- Sector: {info.get('sector', 'N/A')}
- Industry: {info.get('industry', 'N/A')}
- Country: {info.get('country', 'N/A')}
- Website: {info.get('website', 'N/A')}
- Employees: {employees_str}

Business Summary:
{info.get('longBusinessSummary', 'No description available.')[:500]}{'...' if len(info.get('longBusinessSummary', '')) > 500 else ''}"""

        return result

    except Exception as e:
        return f"Error fetching company info for {symbol}: {str(e)}"


# Mock tools for reliability (always work, no external dependencies)

@tool("analyze_stock_performance", "Analyze stock data and provide investment insights")
def analyze_stock_performance(stock_data: str, analysis_type: str = "basic") -> str:
    """Analyze stock performance data and provide insights.

    This is a mock analysis tool that simulates professional analysis.
    In production, this could integrate with real analysis services.

    Args:
        stock_data: Stock information to analyze (from get_stock_quote or get_stock_history)
        analysis_type: Type of analysis - 'basic', 'technical', or 'fundamental'

    Returns:
        Analysis results and insights
    """
    analysis_types = {
        "basic": "Basic Analysis",
        "technical": "Technical Analysis",
        "fundamental": "Fundamental Analysis"
    }

    if analysis_type not in analysis_types:
        analysis_type = "basic"

    # Parse some basic info from the stock data if possible
    lines = stock_data.split('\n')
    symbol = "the stock"
    for line in lines:
        if "Stock Quote for" in line or "Historical Data for" in line:
            parts = line.split('(')
            if len(parts) > 1:
                symbol = parts[1].split(')')[0]
                break

    result = f"""
{analysis_types[analysis_type]} for {symbol}:

Key Observations:
- Market sentiment appears mixed based on recent price action
- Volume patterns suggest moderate institutional interest
- Price is within normal trading range for current market conditions

Risk Assessment:
- Volatility: Moderate
- Liquidity: Good
- Market Risk: Standard for equity investments

Recommendation:
Based on the {analysis_type} analysis, this stock shows typical characteristics
for its sector. Investors should consider their risk tolerance and investment
timeline when making decisions.

Note: This is a simulated analysis for demonstration purposes.
Always conduct thorough research and consider consulting a financial advisor.
"""
    return result.strip()


@tool("generate_investment_report", "Generate a formatted investment report")
def generate_investment_report(
    stocks: str,
    analysis_summary: str,
    report_type: str = "summary"
) -> str:
    """Generate a formatted investment report.

    This is a mock report generator that creates professional-looking reports.

    Args:
        stocks: Comma-separated list of stock symbols covered in the report
        analysis_summary: Summary of analysis findings
        report_type: Type of report - 'summary', 'detailed', or 'comparison'

    Returns:
        Formatted investment report
    """
    from datetime import datetime

    report_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    stock_list = [s.strip().upper() for s in stocks.split(',')]

    if report_type == "comparison":
        report_title = f"Stock Comparison Report: {', '.join(stock_list)}"
    elif report_type == "detailed":
        report_title = f"Detailed Investment Analysis: {', '.join(stock_list)}"
    else:
        report_title = f"Investment Summary Report: {', '.join(stock_list)}"

    result = f"""
{'='*60}
{report_title}
Generated: {report_date}
{'='*60}

EXECUTIVE SUMMARY
-----------------
This report covers {len(stock_list)} stock(s): {', '.join(stock_list)}

{analysis_summary}

MARKET CONTEXT
--------------
Current market conditions show typical volatility patterns.
Sector performance varies, with technology and healthcare
showing relative strength in recent sessions.

KEY TAKEAWAYS
-------------
1. Diversification remains important for risk management
2. Consider both short-term and long-term investment horizons
3. Monitor earnings reports and economic indicators

DISCLAIMER
----------
This report is generated for demonstration purposes only.
It does not constitute financial advice. Always conduct
your own research and consider consulting with a qualified
financial advisor before making investment decisions.

{'='*60}
End of Report
{'='*60}
"""
    return result.strip()


__all__ = [
    'get_stock_quote',
    'get_stock_history',
    'get_company_info',
    'analyze_stock_performance',
    'generate_investment_report',
]
