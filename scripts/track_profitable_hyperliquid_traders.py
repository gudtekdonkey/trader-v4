import asyncio
import aiohttp
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

DATA_DIR = '../data'
PARENT_DIR = '/traders'
FILE_NAME = '/profitable_traders.json'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Data class for trade information"""
    trader_address: str
    symbol: str
    side: str  # "buy" or "sell"
    size: float
    price: float
    leverage: float
    timestamp: str
    trade_id: str
    pnl: Optional[float] = None
    
@dataclass
class Position:
    """Data class for position information"""
    trader_address: str
    symbol: str
    size: float
    entry_price: float
    current_price: float
    leverage: float
    unrealized_pnl: float
    margin_used: float
    timestamp: str

class RealTimeTradeTracker:
    def __init__(self, profitable_traders_file: str = f"{DATA_DIR}{PARENT_DIR}{FILE_NAME}"):
        self.base_url = "https://api.hyperliquid.xyz/info"
        self.traders_to_monitor = self.load_profitable_traders(profitable_traders_file)
        self.known_trades: Dict[str, Set[str]] = defaultdict(set)  # trader -> trade_ids
        self.active_positions: Dict[str, List[Position]] = defaultdict(list)
        self.trade_log_file = "realtime_trades.json"
        self.position_log_file = "active_positions.json"
        self.alert_threshold = 10000  # Alert for trades > $10k
        
    def load_profitable_traders(self, filename: str) -> List[str]:
        """Load the list of profitable traders to monitor"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                traders = list(data.get("traders", {}).keys())
                logger.info(f"Loaded {len(traders)} profitable traders to monitor")
                return traders
        except FileNotFoundError:
            logger.error(f"File {filename} not found. Please run the trader tracker first.")
            return []
        except Exception as e:
            logger.error(f"Error loading traders: {e}")
            return []
    
    async def fetch_data(self, session: aiohttp.ClientSession, payload: dict) -> Optional[dict]:
        """Fetch data from Hyperliquid API"""
        try:
            async with session.post(self.base_url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"API error: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Fetch error: {e}")
            return None
    
    async def get_recent_trades(self, session: aiohttp.ClientSession, trader_address: str) -> List[Trade]:
        """Get recent trades for a specific trader"""
        payload = {
            "type": "userFills",
            "user": trader_address
        }
        
        data = await self.fetch_data(session, payload)
        if not data:
            return []
        
        trades = []
        for trade_data in data[-20:]:  # Last 20 trades
            trade = Trade(
                trader_address=trader_address,
                symbol=trade_data.get("coin", ""),
                side=trade_data.get("side", ""),
                size=float(trade_data.get("sz", 0)),
                price=float(trade_data.get("px", 0)),
                leverage=float(trade_data.get("leverage", 1)),
                timestamp=trade_data.get("time", ""),
                trade_id=trade_data.get("tid", ""),
                pnl=float(trade_data.get("closedPnl", 0)) if trade_data.get("closedPnl") else None
            )
            trades.append(trade)
        
        return trades
    
    async def get_positions(self, session: aiohttp.ClientSession, trader_address: str) -> List[Position]:
        """Get current positions for a trader"""
        payload = {
            "type": "clearinghouseState",
            "user": trader_address
        }
        
        data = await self.fetch_data(session, payload)
        if not data:
            return []
        
        positions = []
        asset_positions = data.get("assetPositions", [])
        
        for pos_data in asset_positions:
            position = pos_data.get("position", {})
            if position.get("szi", 0) != 0:  # Has an open position
                pos = Position(
                    trader_address=trader_address,
                    symbol=position.get("coin", ""),
                    size=float(position.get("szi", 0)),
                    entry_price=float(position.get("entryPx", 0)),
                    current_price=float(position.get("markPx", 0)),
                    leverage=float(position.get("leverage", 1)),
                    unrealized_pnl=float(position.get("unrealizedPnl", 0)),
                    margin_used=float(position.get("marginUsed", 0)),
                    timestamp=datetime.now().isoformat()
                )
                positions.append(pos)
        
        return positions
    
    def detect_new_trades(self, trader: str, trades: List[Trade]) -> List[Trade]:
        """Detect new trades that haven't been seen before"""
        new_trades = []
        
        for trade in trades:
            if trade.trade_id not in self.known_trades[trader]:
                self.known_trades[trader].add(trade.trade_id)
                new_trades.append(trade)
        
        return new_trades
    
    def format_trade_alert(self, trade: Trade) -> str:
        """Format a trade alert message"""
        trade_value = trade.size * trade.price
        emoji = "🟢" if trade.side.lower() == "buy" else "🔴"
        
        alert = f"""
{emoji} NEW TRADE ALERT {emoji}
Trader: {trade.trader_address[:10]}...
Symbol: {trade.symbol}
Side: {trade.side.upper()}
Size: {trade.size}
Price: ${trade.price:,.2f}
Value: ${trade_value:,.2f}
Leverage: {trade.leverage}x
Time: {trade.timestamp}
"""
        
        if trade.pnl is not None:
            alert += f"Closed PnL: ${trade.pnl:,.2f}\n"
        
        return alert
    
    def format_position_update(self, position: Position) -> str:
        """Format a position update message"""
        pnl_emoji = "📈" if position.unrealized_pnl > 0 else "📉"
        
        update = f"""
{pnl_emoji} POSITION UPDATE
Trader: {position.trader_address[:10]}...
Symbol: {position.symbol}
Size: {position.size}
Entry: ${position.entry_price:,.2f}
Current: ${position.current_price:,.2f}
Unrealized PnL: ${position.unrealized_pnl:,.2f}
Leverage: {position.leverage}x
Margin Used: ${position.margin_used:,.2f}
"""
        return update
    
    def save_trade_log(self, trades: List[Trade]):
        """Save trades to log file"""
        try:
            # Load existing trades
            try:
                with open(self.trade_log_file, 'r') as f:
                    existing_data = json.load(f)
            except FileNotFoundError:
                existing_data = {"trades": []}
            
            # Add new trades
            for trade in trades:
                existing_data["trades"].append(asdict(trade))
            
            # Save updated data
            with open(self.trade_log_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error saving trade log: {e}")
    
    def save_position_snapshot(self):
        """Save current positions snapshot"""
        try:
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "positions": {}
            }
            
            for trader, positions in self.active_positions.items():
                snapshot["positions"][trader] = [asdict(pos) for pos in positions]
            
            with open(self.position_log_file, 'w') as f:
                json.dump(snapshot, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error saving position snapshot: {e}")
    
    async def monitor_trader(self, session: aiohttp.ClientSession, trader: str):
        """Monitor a single trader for new trades and position changes"""
        try:
            # Get recent trades
            trades = await self.get_recent_trades(session, trader)
            new_trades = self.detect_new_trades(trader, trades)
            
            # Alert on new trades
            if new_trades:
                for trade in new_trades:
                    trade_value = trade.size * trade.price
                    
                    # Log all trades
                    logger.info(f"New trade: {trade.symbol} {trade.side} ${trade_value:,.2f}")
                    
                    # Alert for large trades
                    if trade_value > self.alert_threshold:
                        logger.warning(f"LARGE TRADE ALERT!")
                        print(self.format_trade_alert(trade))
                
                # Save new trades
                self.save_trade_log(new_trades)
            
            # Get current positions
            positions = await self.get_positions(session, trader)
            self.active_positions[trader] = positions
            
            # Alert on significant position changes
            for position in positions:
                if abs(position.unrealized_pnl) > 1000:  # Alert for PnL > $1000
                    logger.info(f"Significant position: {position.symbol} PnL: ${position.unrealized_pnl:,.2f}")
        
        except Exception as e:
            logger.error(f"Error monitoring trader {trader}: {e}")
    
    async def monitor_all_traders(self, session: aiohttp.ClientSession):
        """Monitor all profitable traders"""
        tasks = []
        
        for trader in self.traders_to_monitor:
            task = self.monitor_trader(session, trader)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def run_monitoring_loop(self, interval: int = 10):
        """Main monitoring loop"""
        logger.info(f"Starting real-time monitoring of {len(self.traders_to_monitor)} traders")
        logger.info(f"Update interval: {interval} seconds")
        
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    start_time = time.time()
                    
                    # Monitor all traders
                    await self.monitor_all_traders(session)
                    
                    # Save position snapshot
                    self.save_position_snapshot()
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    logger.debug(f"Processing completed in {processing_time:.2f} seconds")
                    
                    # Wait for next iteration
                    await asyncio.sleep(max(0, interval - processing_time))
                
                except KeyboardInterrupt:
                    logger.info("Monitoring stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(interval)
    
    def get_statistics(self) -> Dict:
        """Get current monitoring statistics"""
        stats = {
            "traders_monitored": len(self.traders_to_monitor),
            "total_positions": sum(len(pos) for pos in self.active_positions.values()),
            "total_trades_tracked": sum(len(trades) for trades in self.known_trades.values()),
            "traders_with_positions": len([t for t, p in self.active_positions.items() if p])
        }
        return stats

# Alert configuration
class AlertConfig:
    def __init__(self):
        self.trade_size_threshold = 10000  # Alert for trades > $10k
        self.pnl_threshold = 1000  # Alert for PnL > $1k
        self.leverage_threshold = 10  # Alert for leverage > 10x
        self.enable_sound = False  # Enable sound alerts
        self.webhook_url = None  # Discord/Telegram webhook URL

# Main execution
async def main():
    # Configuration
    config = {
        "update_interval": 10,  # seconds
        "traders_file": f"{DATA_DIR}{PARENT_DIR}{FILE_NAME}"
    }
    
    # Initialize tracker
    tracker = RealTimeTradeTracker(config["traders_file"])
    
    if not tracker.traders_to_monitor:
        logger.error("No traders to monitor. Please run the profitable trader finder first.")
        return
    
    # Start monitoring
    await tracker.run_monitoring_loop(interval=config["update_interval"])

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Monitoring stopped.")