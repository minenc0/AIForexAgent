"""Risk Management Agent — calculates entry, SL, TP, and position size."""

from __future__ import annotations

from dataclasses import dataclass

from config import Decision, RiskConfig, Timeframe
from data.technical import compute_technical_analysis
from utils.logger import logger
from utils.risk import RiskResult, calculate_risk


@dataclass
class RiskAgentResult:
    """Result of the risk management analysis."""

    entry: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    risk_pips: float
    reward_pips: float
    lot_size: float
    risk_amount: float
    reward_amount: float
    details: str
    raw: Optional[RiskResult] = None


class RiskAgent:
    """Calculates risk parameters for the trade.

    Uses ATR-based dynamic positioning with a fixed risk/reward ratio
    of 1:2 (0.17% risk, 0.34% reward per trade).
    """

    def __init__(
        self,
        risk_config: Optional[RiskConfig] = None,
        balance: float = 10000.0,
    ) -> None:
        """Initialise the RiskAgent.

        Args:
            risk_config: Custom risk parameters. Uses defaults if ``None``.
            balance: Account balance for position sizing.
        """
        self.risk_config = risk_config or RiskConfig()
        self.balance = balance

    def analyse(
        self,
        pair: str,
        timeframe: Timeframe,
        decision: Decision,
    ) -> RiskAgentResult:
        """Calculate risk management parameters.

        Args:
            pair: Forex pair string.
            timeframe: Bar period.
            decision: The trade direction decision.

        Returns:
            A ``RiskAgentResult`` with all risk parameters.
        """
        logger.info("RiskAgent analysing %s @ %s (decision=%s)", pair, timeframe.value, decision)

        # Get current price and ATR
        try:
            ta = compute_technical_analysis(pair, timeframe)
            current_price = ta.current_price
            atr_value = ta.atr14 or current_price * 0.001
        except (ValueError, Exception) as e:
            logger.error("RiskAgent: failed to get data: %s", e)
            return RiskAgentResult(
                entry=0.0, stop_loss=0.0, take_profit=0.0,
                risk_reward=0.0, risk_pips=0.0, reward_pips=0.0,
                lot_size=0.0, risk_amount=0.0, reward_amount=0.0,
                details=f"Data unavailable: {e}",
            )

        risk_result = calculate_risk(
            current_price=current_price,
            atr_value=atr_value,
            decision=decision,
            risk_config=self.risk_config,
            balance=self.balance,
        )

        details = (
            f"Entry: {risk_result.entry:.5f} | "
            f"SL: {risk_result.stop_loss:.5f} | "
            f"TP: {risk_result.take_profit:.5f} | "
            f"R:R = 1:{risk_result.risk_reward:.1f} | "
            f"Risk: {risk_result.risk_amount:.2f} | "
            f"Reward: {risk_result.reward_amount:.2f}"
        )

        logger.info("RiskAgent result: %s", details)

        return RiskAgentResult(
            entry=risk_result.entry,
            stop_loss=risk_result.stop_loss,
            take_profit=risk_result.take_profit,
            risk_reward=risk_result.risk_reward,
            risk_pips=risk_result.risk_pips,
            reward_pips=risk_result.reward_pips,
            lot_size=risk_result.lot_size,
            risk_amount=risk_result.risk_amount,
            reward_amount=risk_result.reward_amount,
            details=details,
            raw=risk_result,
        )