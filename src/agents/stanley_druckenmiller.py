from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import (
    get_financial_metrics,
    get_market_cap,
    search_line_items,
    get_insider_trades,
    get_company_news,
    get_prices,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm
import statistics
from src.utils.api_key import get_api_key_from_state

class StanleyDruckenmillerSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def stanley_druckenmiller_agent(state: AgentState, agent_id: str = "stanley_druckenmiller_agent"):
    """
    Analyzes global macro ETFs using Stanley Druckenmiller's macro investing principles:
      - Focus on macroeconomic trends and policy shifts
      - Analyze currency movements, interest rates, and commodity cycles
      - Seek asymmetric risk-reward opportunities in macro themes
      - Emphasize top-down analysis of economic indicators
      - Willing to be aggressive on macro convictions
      - Focus on preserving capital through macro risk management

    Returns a bullish/bearish/neutral signal with confidence and reasoning.
    """
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    etfs = data["etfs"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    analysis_data = {}
    druck_analysis = {}

    for etf in etfs:
        progress.update_status(agent_id, etf, "Fetching financial metrics")
        metrics = get_financial_metrics(etf, end_date, period="annual", limit=5, api_key=api_key)

        progress.update_status(agent_id, etf, "Gathering financial line items")
        # Include relevant line items for Stan Druckenmiller's approach:
        #   - Growth & momentum: revenue, EPS, operating_income, ...
        #   - Valuation: net_income, free_cash_flow, ebit, ebitda
        #   - Leverage: total_debt, shareholders_equity
        #   - Liquidity: cash_and_equivalents
        financial_line_items = search_line_items(
            etf,
            [
                "revenue",
                "earnings_per_share",
                "net_income",
                "operating_income",
                "gross_margin",
                "operating_margin",
                "free_cash_flow",
                "capital_expenditure",
                "cash_and_equivalents",
                "total_debt",
                "shareholders_equity",
                "outstanding_shares",
                "ebit",
                "ebitda",
            ],
            end_date,
            period="annual",
            limit=5,
            api_key=api_key,
        )

        progress.update_status(agent_id, etf, "Getting market cap")
        market_cap = get_market_cap(etf, end_date, api_key=api_key)

        progress.update_status(agent_id, etf, "Fetching insider trades")
        insider_trades = get_insider_trades(etf, end_date, limit=50, api_key=api_key)

        progress.update_status(agent_id, etf, "Fetching company news")
        company_news = get_company_news(etf, end_date, limit=50, api_key=api_key)

        progress.update_status(agent_id, etf, "Fetching recent price data for momentum")
        prices = get_prices(etf, start_date=start_date, end_date=end_date, api_key=api_key)

        progress.update_status(agent_id, etf, "Analyzing macro trends")
        macro_trends_analysis = analyze_macro_trends(financial_line_items, prices)

        progress.update_status(agent_id, etf, "Analyzing economic indicators")
        economic_indicators_analysis = analyze_economic_indicators(financial_line_items, company_news)

        progress.update_status(agent_id, etf, "Analyzing policy environment")
        policy_analysis = analyze_policy_environment(company_news, insider_trades)

        progress.update_status(agent_id, etf, "Analyzing macro risk-reward")
        macro_risk_reward_analysis = analyze_macro_risk_reward(financial_line_items, prices)

        progress.update_status(agent_id, etf, "Performing macro valuation")
        macro_valuation_analysis = analyze_macro_valuation(financial_line_items, market_cap)

        # Combine partial scores with weights typical for Druckenmiller macro analysis:
        #   35% Macro Trends, 25% Economic Indicators, 20% Policy Environment,
        #   15% Macro Risk/Reward, 5% Macro Valuation = 100%
        total_score = (
            macro_trends_analysis["score"] * 0.35
            + economic_indicators_analysis["score"] * 0.25
            + policy_analysis["score"] * 0.20
            + macro_risk_reward_analysis["score"] * 0.15
            + macro_valuation_analysis["score"] * 0.05
        )

        max_possible_score = 10

        # Simple bullish/neutral/bearish signal
        if total_score >= 7.5:
            signal = "bullish"
        elif total_score <= 4.5:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[etf] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "macro_trends_analysis": macro_trends_analysis,
            "economic_indicators_analysis": economic_indicators_analysis,
            "policy_analysis": policy_analysis,
            "macro_risk_reward_analysis": macro_risk_reward_analysis,
            "macro_valuation_analysis": macro_valuation_analysis,
        }

        progress.update_status(agent_id, etf, "Generating Stanley Druckenmiller analysis")
        druck_output = generate_druckenmiller_output(
            etf=etf,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        druck_analysis[etf] = {
            "signal": druck_output.signal,
            "confidence": druck_output.confidence,
            "reasoning": druck_output.reasoning,
        }

        progress.update_status(agent_id, etf, "Done", analysis=druck_output.reasoning)

    # Wrap results in a single message
    message = HumanMessage(content=json.dumps(druck_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(druck_analysis, "Stanley Druckenmiller Agent")

    state["data"]["analyst_signals"][agent_id] = druck_analysis

    progress.update_status(agent_id, None, "Done")
    
    return {"messages": [message], "data": state["data"]}


def analyze_macro_trends(financial_line_items: list, prices: list) -> dict:
    """
    Evaluate macro trends affecting the ETF:
      - Economic growth indicators
      - Inflation trends
      - Interest rate environment
      - Currency strength
      - Commodity cycles
    """
    # Placeholder for macro trend analysis
    # In a real implementation, this would analyze:
    # - GDP growth rates
    # - Inflation data
    # - Interest rate trends
    # - Currency movements
    # - Commodity price cycles
    
    details = ["Macro trend analysis - placeholder for economic indicators"]
    raw_score = 5  # Neutral score for now
    
    # Scale to 0-10
    score = min(10, max(0, raw_score))
    
    return {
        "score": score,
        "details": details,
        "macro_indicators": "Placeholder for GDP, inflation, rates analysis"
    }

    #
    # 1. Revenue Growth (annualized CAGR)
    #
    revenues = [fi.revenue for fi in financial_line_items if fi.revenue is not None]
    if len(revenues) >= 2:
        latest_rev = revenues[0]
        older_rev = revenues[-1]
        num_years = len(revenues) - 1
        if older_rev > 0 and latest_rev > 0:
            # CAGR formula: (ending_value/beginning_value)^(1/years) - 1
            rev_growth = (latest_rev / older_rev) ** (1 / num_years) - 1
            if rev_growth > 0.08:  # 8% annualized (adjusted for CAGR)
                raw_score += 3
                details.append(f"Strong annualized revenue growth: {rev_growth:.1%}")
            elif rev_growth > 0.04:  # 4% annualized
                raw_score += 2
                details.append(f"Moderate annualized revenue growth: {rev_growth:.1%}")
            elif rev_growth > 0.01:  # 1% annualized
                raw_score += 1
                details.append(f"Slight annualized revenue growth: {rev_growth:.1%}")
            else:
                details.append(f"Minimal/negative revenue growth: {rev_growth:.1%}")
        else:
            details.append("Older revenue is zero/negative; can't compute revenue growth.")
    else:
        details.append("Not enough revenue data points for growth calculation.")

    #
    # 2. EPS Growth (annualized CAGR)
    #
    eps_values = [fi.earnings_per_share for fi in financial_line_items if fi.earnings_per_share is not None]
    if len(eps_values) >= 2:
        latest_eps = eps_values[0]
        older_eps = eps_values[-1]
        num_years = len(eps_values) - 1
        # Calculate CAGR for positive EPS values
        if older_eps > 0 and latest_eps > 0:
            # CAGR formula for EPS
            eps_growth = (latest_eps / older_eps) ** (1 / num_years) - 1
            if eps_growth > 0.08:  # 8% annualized (adjusted for CAGR)
                raw_score += 3
                details.append(f"Strong annualized EPS growth: {eps_growth:.1%}")
            elif eps_growth > 0.04:  # 4% annualized
                raw_score += 2
                details.append(f"Moderate annualized EPS growth: {eps_growth:.1%}")
            elif eps_growth > 0.01:  # 1% annualized
                raw_score += 1
                details.append(f"Slight annualized EPS growth: {eps_growth:.1%}")
            else:
                details.append(f"Minimal/negative annualized EPS growth: {eps_growth:.1%}")
        else:
            details.append("Older EPS is near zero; skipping EPS growth calculation.")
    else:
        details.append("Not enough EPS data points for growth calculation.")

    #
    # 3. Price Momentum
    #
    # We'll give up to 3 points for strong momentum
    if prices and len(prices) > 30:
        sorted_prices = sorted(prices, key=lambda p: p.time)
        close_prices = [p.close for p in sorted_prices if p.close is not None]
        if len(close_prices) >= 2:
            start_price = close_prices[0]
            end_price = close_prices[-1]
            if start_price > 0:
                pct_change = (end_price - start_price) / start_price
                if pct_change > 0.50:
                    raw_score += 3
                    details.append(f"Very strong price momentum: {pct_change:.1%}")
                elif pct_change > 0.20:
                    raw_score += 2
                    details.append(f"Moderate price momentum: {pct_change:.1%}")
                elif pct_change > 0:
                    raw_score += 1
                    details.append(f"Slight positive momentum: {pct_change:.1%}")
                else:
                    details.append(f"Negative price momentum: {pct_change:.1%}")
            else:
                details.append("Invalid start price (<= 0); can't compute momentum.")
        else:
            details.append("Insufficient price data for momentum calculation.")
    else:
        details.append("Not enough recent price data for momentum analysis.")

    # We assigned up to 3 points each for:
    #   revenue growth, eps growth, momentum
    # => max raw_score = 9
    # Scale to 0–10
    final_score = min(10, (raw_score / 9) * 10)

    return {"score": final_score, "details": "; ".join(details)}


def analyze_economic_indicators(financial_line_items: list, company_news: list) -> dict:
    """
    Analyze economic indicators affecting the ETF:
      - GDP growth rates
      - Employment data
      - Consumer spending
      - Manufacturing indicators
    """
    # Default is neutral (5/10).
    score = 5
    details = []

    if not insider_trades:
        details.append("No insider trades data; defaulting to neutral")
        return {"score": score, "details": "; ".join(details)}

    buys, sells = 0, 0
    for trade in insider_trades:
        # Use transaction_shares to determine if it's a buy or sell
        # Negative shares = sell, positive shares = buy
        if trade.transaction_shares is not None:
            if trade.transaction_shares > 0:
                buys += 1
            elif trade.transaction_shares < 0:
                sells += 1

    total = buys + sells
    if total == 0:
        details.append("No buy/sell transactions found; neutral")
        return {"score": score, "details": "; ".join(details)}

    buy_ratio = buys / total
    if buy_ratio > 0.7:
        # Heavy buying => +3 points from the neutral 5 => 8
        score = 8
        details.append(f"Heavy insider buying: {buys} buys vs. {sells} sells")
    elif buy_ratio > 0.4:
        # Moderate buying => +1 => 6
        score = 6
        details.append(f"Moderate insider buying: {buys} buys vs. {sells} sells")
    else:
        # Low insider buying => -1 => 4
        score = 4
        details.append(f"Mostly insider selling: {buys} buys vs. {sells} sells")

    return {"score": score, "details": "; ".join(details)}


def analyze_sentiment(news_items: list) -> dict:
    """
    Basic news sentiment: negative keyword check vs. overall volume.
    """
    if not news_items:
        return {"score": 5, "details": "No news data; defaulting to neutral sentiment"}

    negative_keywords = ["lawsuit", "fraud", "negative", "downturn", "decline", "investigation", "recall"]
    negative_count = 0
    for news in news_items:
        title_lower = (news.title or "").lower()
        if any(word in title_lower for word in negative_keywords):
            negative_count += 1

    details = []
    if negative_count > len(news_items) * 0.3:
        # More than 30% negative => somewhat bearish => 3/10
        score = 3
        details.append(f"High proportion of negative headlines: {negative_count}/{len(news_items)}")
    elif negative_count > 0:
        # Some negativity => 6/10
        score = 6
        details.append(f"Some negative headlines: {negative_count}/{len(news_items)}")
    else:
        # Mostly positive => 8/10
        score = 8
        details.append("Mostly positive/neutral headlines")

    return {"score": score, "details": "; ".join(details)}


def analyze_risk_reward(financial_line_items: list, prices: list) -> dict:
    """
    Assesses risk via:
      - Debt-to-Equity
      - Price Volatility
    Aims for strong upside with contained downside.
    """
    if not financial_line_items or not prices:
        return {"score": 0, "details": "Insufficient data for risk-reward analysis"}

    details = []
    raw_score = 0  # We'll accumulate up to 6 raw points, then scale to 0-10

    #
    # 1. Debt-to-Equity
    #
    debt_values = [fi.total_debt for fi in financial_line_items if fi.total_debt is not None]
    equity_values = [fi.shareholders_equity for fi in financial_line_items if fi.shareholders_equity is not None]

    if debt_values and equity_values and len(debt_values) == len(equity_values) and len(debt_values) > 0:
        recent_debt = debt_values[0]
        recent_equity = equity_values[0] if equity_values[0] else 1e-9
        de_ratio = recent_debt / recent_equity
        if de_ratio < 0.3:
            raw_score += 3
            details.append(f"Low debt-to-equity: {de_ratio:.2f}")
        elif de_ratio < 0.7:
            raw_score += 2
            details.append(f"Moderate debt-to-equity: {de_ratio:.2f}")
        elif de_ratio < 1.5:
            raw_score += 1
            details.append(f"Somewhat high debt-to-equity: {de_ratio:.2f}")
        else:
            details.append(f"High debt-to-equity: {de_ratio:.2f}")
    else:
        details.append("No consistent debt/equity data available.")

    #
    # 2. Price Volatility
    #
    if len(prices) > 10:
        sorted_prices = sorted(prices, key=lambda p: p.time)
        close_prices = [p.close for p in sorted_prices if p.close is not None]
        if len(close_prices) > 10:
            daily_returns = []
            for i in range(1, len(close_prices)):
                prev_close = close_prices[i - 1]
                if prev_close > 0:
                    daily_returns.append((close_prices[i] - prev_close) / prev_close)
            if daily_returns:
                stdev = statistics.pstdev(daily_returns)  # population stdev
                if stdev < 0.01:
                    raw_score += 3
                    details.append(f"Low volatility: daily returns stdev {stdev:.2%}")
                elif stdev < 0.02:
                    raw_score += 2
                    details.append(f"Moderate volatility: daily returns stdev {stdev:.2%}")
                elif stdev < 0.04:
                    raw_score += 1
                    details.append(f"High volatility: daily returns stdev {stdev:.2%}")
                else:
                    details.append(f"Very high volatility: daily returns stdev {stdev:.2%}")
            else:
                details.append("Insufficient daily returns data for volatility calc.")
        else:
            details.append("Not enough close-price data points for volatility analysis.")
    else:
        details.append("Not enough price data for volatility analysis.")

    # raw_score out of 6 => scale to 0–10
    final_score = min(10, (raw_score / 6) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_druckenmiller_valuation(financial_line_items: list, market_cap: float | None) -> dict:
    """
    Druckenmiller is willing to pay up for growth, but still checks:
      - P/E
      - P/FCF
      - EV/EBIT
      - EV/EBITDA
    Each can yield up to 2 points => max 8 raw points => scale to 0–10.
    """
    if not financial_line_items or market_cap is None:
        return {"score": 0, "details": "Insufficient data to perform valuation"}

    details = []
    raw_score = 0

    # Gather needed data
    net_incomes = [fi.net_income for fi in financial_line_items if fi.net_income is not None]
    fcf_values = [fi.free_cash_flow for fi in financial_line_items if fi.free_cash_flow is not None]
    ebit_values = [fi.ebit for fi in financial_line_items if fi.ebit is not None]
    ebitda_values = [fi.ebitda for fi in financial_line_items if fi.ebitda is not None]

    # For EV calculation, let's get the most recent total_debt & cash
    debt_values = [fi.total_debt for fi in financial_line_items if fi.total_debt is not None]
    cash_values = [fi.cash_and_equivalents for fi in financial_line_items if fi.cash_and_equivalents is not None]
    recent_debt = debt_values[0] if debt_values else 0
    recent_cash = cash_values[0] if cash_values else 0

    enterprise_value = market_cap + recent_debt - recent_cash

    # 1) P/E
    recent_net_income = net_incomes[0] if net_incomes else None
    if recent_net_income and recent_net_income > 0:
        pe = market_cap / recent_net_income
        pe_points = 0
        if pe < 15:
            pe_points = 2
            details.append(f"Attractive P/E: {pe:.2f}")
        elif pe < 25:
            pe_points = 1
            details.append(f"Fair P/E: {pe:.2f}")
        else:
            details.append(f"High or Very high P/E: {pe:.2f}")
        raw_score += pe_points
    else:
        details.append("No positive net income for P/E calculation")

    # 2) P/FCF
    recent_fcf = fcf_values[0] if fcf_values else None
    if recent_fcf and recent_fcf > 0:
        pfcf = market_cap / recent_fcf
        pfcf_points = 0
        if pfcf < 15:
            pfcf_points = 2
            details.append(f"Attractive P/FCF: {pfcf:.2f}")
        elif pfcf < 25:
            pfcf_points = 1
            details.append(f"Fair P/FCF: {pfcf:.2f}")
        else:
            details.append(f"High/Very high P/FCF: {pfcf:.2f}")
        raw_score += pfcf_points
    else:
        details.append("No positive free cash flow for P/FCF calculation")

    # 3) EV/EBIT
    recent_ebit = ebit_values[0] if ebit_values else None
    if enterprise_value > 0 and recent_ebit and recent_ebit > 0:
        ev_ebit = enterprise_value / recent_ebit
        ev_ebit_points = 0
        if ev_ebit < 15:
            ev_ebit_points = 2
            details.append(f"Attractive EV/EBIT: {ev_ebit:.2f}")
        elif ev_ebit < 25:
            ev_ebit_points = 1
            details.append(f"Fair EV/EBIT: {ev_ebit:.2f}")
        else:
            details.append(f"High EV/EBIT: {ev_ebit:.2f}")
        raw_score += ev_ebit_points
    else:
        details.append("No valid EV/EBIT because EV <= 0 or EBIT <= 0")

    # 4) EV/EBITDA
    recent_ebitda = ebitda_values[0] if ebitda_values else None
    if enterprise_value > 0 and recent_ebitda and recent_ebitda > 0:
        ev_ebitda = enterprise_value / recent_ebitda
        ev_ebitda_points = 0
        if ev_ebitda < 10:
            ev_ebitda_points = 2
            details.append(f"Attractive EV/EBITDA: {ev_ebitda:.2f}")
        elif ev_ebitda < 18:
            ev_ebitda_points = 1
            details.append(f"Fair EV/EBITDA: {ev_ebitda:.2f}")
        else:
            details.append(f"High EV/EBITDA: {ev_ebitda:.2f}")
        raw_score += ev_ebitda_points
    else:
        details.append("No valid EV/EBITDA because EV <= 0 or EBITDA <= 0")

    # We have up to 2 points for each of the 4 metrics => 8 raw points max
    # Scale raw_score to 0–10
    final_score = min(10, (raw_score / 8) * 10)

    return {"score": final_score, "details": "; ".join(details)}


def generate_druckenmiller_output(
    etf: str,
    analysis_data: dict[str, any],
    state: AgentState,
    agent_id: str,
) -> StanleyDruckenmillerSignal:
    """
    Generates a JSON signal in the style of Stanley Druckenmiller.
    """
    template = ChatPromptTemplate.from_messages(
        [
            (
              "system",
              """You are a Stanley Druckenmiller AI agent, making macro investment decisions using his principles:
            
              1. Focus on macroeconomic trends and policy shifts that drive markets.
              2. Analyze currency movements, interest rates, and commodity cycles.
              3. Seek asymmetric risk-reward opportunities in macro themes.
              4. Emphasize top-down analysis of economic indicators.
              5. Be aggressive when macro conviction is high.
              6. Cut losses quickly if the macro thesis changes.
                            
              Rules:
              - Analyze global macro trends affecting the ETF's underlying assets.
              - Evaluate economic indicators, central bank policies, and geopolitical factors.
              - Consider currency strength, interest rate cycles, and commodity trends.
              - Assess risk-reward from a macro perspective with specific economic data.
              - Output a JSON object with signal, confidence, and a reasoning string.
              
              When providing your reasoning, be thorough and specific by:
              1. Explaining the macro trends and economic indicators that most influenced your decision
              2. Highlighting the macro risk-reward profile with specific economic evidence
              3. Discussing policy changes and catalysts that could drive macro performance
              4. Addressing both upside potential and downside risks from a macro perspective
              5. Providing specific context about economic cycles and policy impacts
              6. Using Stanley Druckenmiller's decisive, macro-focused, and conviction-driven voice
              
              For example, if bullish: "The ETF benefits from strong macro tailwinds with GDP growth accelerating to 3.2% and inflation moderating to 2.1%. The Fed's dovish pivot suggests rate cuts ahead, which historically drives this asset class 25-40% higher. Risk-reward is highly asymmetric with 60% upside potential based on historical macro cycles and only 15% downside risk given the supportive policy environment. The currency backdrop is favorable with DXY weakening, providing additional tailwinds..."
              For example, if bearish: "Despite recent momentum, the macro environment is deteriorating with leading indicators pointing to recession risk. The Fed's hawkish stance and rising rates create headwinds for this asset class. Risk-reward is unfavorable with limited 10% upside potential against 35% downside risk. The economic cycle is turning, and policy support is waning. I'm seeing better macro opportunities elsewhere with more favorable setups..."
              """,
            ),
            (
              "human",
              """Based on the following macro analysis, create a Druckenmiller-style investment signal.

              Macro Analysis Data for {etf}:
              {analysis_data}

              Return the trading signal in this JSON format:
              {{
                "signal": "bullish/bearish/neutral",
                "confidence": float (0-100),
                "reasoning": "string"
              }}
              """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "etf": etf})

    def create_default_signal():
        return StanleyDruckenmillerSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral"
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=StanleyDruckenmillerSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_signal,
    )


# New macro analysis functions
def analyze_economic_indicators(financial_line_items: list, company_news: list) -> dict:
    """Analyze economic indicators affecting the ETF."""
    return {
        "score": 5,
        "details": ["Economic indicators analysis - placeholder"],
        "indicators": "GDP, employment, consumer spending analysis"
    }


def analyze_policy_environment(company_news: list, insider_trades: list) -> dict:
    """Analyze policy environment and central bank actions."""
    return {
        "score": 5,
        "details": ["Policy environment analysis - placeholder"],
        "policy_indicators": "Central bank policy, fiscal policy analysis"
    }


def analyze_macro_risk_reward(financial_line_items: list, prices: list) -> dict:
    """Analyze macro risk-reward profile."""
    return {
        "score": 5,
        "details": ["Macro risk-reward analysis - placeholder"],
        "risk_factors": "Economic cycle, policy risk, geopolitical risk"
    }


def analyze_macro_valuation(financial_line_items: list, market_cap: float | None) -> dict:
    """Analyze macro valuation metrics."""
    return {
        "score": 5,
        "details": ["Macro valuation analysis - placeholder"],
        "valuation_metrics": "Relative valuation, macro multiples"
    }
