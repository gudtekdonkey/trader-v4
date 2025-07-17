import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio
import aiohttp
from collections import defaultdict

DATA_DIR = '../data'
PARENT_DIR = '/traders'
FILE_NAME = '/profitable_traders.json'

class HyperliquidTraderTracker:

    def __init__(self):
        self.base_url = "https://api.hyperliquid.xyz/info"
        self.profitable_traders = {}
        self.output_file = f"{DATA_DIR}{PARENT_DIR}{FILE_NAME}"
        
    async def fetch_data(self, session: aiohttp.ClientSession, endpoint: str, payload: dict) -> Optional[dict]:
        """Async fetch data from Hyperliquid API"""
        try:
            async with session.post(f"{self.base_url}", json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Error fetching data: {response.status}")
                    return None
        except Exception as e:
            print(f"Error in fetch_data: {e}")
            return None
    
    async def get_recent_trades(self, session: aiohttp.ClientSession) -> List[dict]:
        """Get recent trades from the API"""
        payload = {
            "type": "allMids"
        }
        data = await self.fetch_data(session, "", payload)
        return data if data else []
    
    async def get_user_trades(self, session: aiohttp.ClientSession, user_address: str) -> List[dict]:
        """Get trades for a specific user"""
        payload = {
            "type": "userFills",
            "user": user_address
        }
        data = await self.fetch_data(session, "", payload)
        return data if data else []
    
    async def get_user_positions(self, session: aiohttp.ClientSession, user_address: str) -> List[dict]:
        """Get current positions for a user"""
        payload = {
            "type": "clearinghouseState",
            "user": user_address
        }
        data = await self.fetch_data(session, "", payload)
        return data.get("assetPositions", []) if data else []
    
    async def get_leaderboard(self, session: aiohttp.ClientSession) -> List[dict]:
        """Get the current leaderboard data"""
        payload = {
            "type": "leaderboard",
            "timeWindow": "day"  # Can be "day", "week", "month", "allTime"
        }
        data = await self.fetch_data(session, "", payload)
        return data if data else []
    
    def calculate_trader_metrics(self, trades: List[dict], positions: List[dict]) -> Dict:
        """Calculate profitability metrics for a trader"""
        metrics = {
            "total_pnl": 0,
            "win_rate": 0,
            "total_trades": len(trades),
            "winning_trades": 0,
            "losing_trades": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "uses_margin": False,
            "max_leverage": 0,
            "current_positions": len(positions)
        }
        
        if not trades:
            return metrics
        
        wins = []
        losses = []
        
        for trade in trades:
            # Check if trade uses margin (leverage > 1)
            leverage = trade.get("leverage", 1)
            if leverage > 1:
                metrics["uses_margin"] = True
                metrics["max_leverage"] = max(metrics["max_leverage"], leverage)
            
            # Calculate PnL for the trade
            pnl = trade.get("pnl", 0)
            metrics["total_pnl"] += pnl
            
            if pnl > 0:
                metrics["winning_trades"] += 1
                wins.append(pnl)
            elif pnl < 0:
                metrics["losing_trades"] += 1
                losses.append(abs(pnl))
        
        # Calculate averages and ratios
        if wins:
            metrics["avg_win"] = sum(wins) / len(wins)
        if losses:
            metrics["avg_loss"] = sum(losses) / len(losses)
        
        if metrics["total_trades"] > 0:
            metrics["win_rate"] = metrics["winning_trades"] / metrics["total_trades"]
        
        if metrics["avg_loss"] > 0:
            metrics["profit_factor"] = metrics["avg_win"] / metrics["avg_loss"]
        
        return metrics
    
    async def analyze_traders(self, session: aiohttp.ClientSession, trader_addresses: List[str]) -> Dict:
        """Analyze multiple traders and their profitability"""
        trader_data = {}
        
        for address in trader_addresses:
            print(f"Analyzing trader: {address}")
            
            # Get trader's trades and positions
            trades = await self.get_user_trades(session, address)
            positions = await self.get_user_positions(session, address)
            
            # Calculate metrics
            metrics = self.calculate_trader_metrics(trades, positions)
            
            # Only include traders who use margin
            if metrics["uses_margin"]:
                trader_data[address] = {
                    "address": address,
                    "metrics": metrics,
                    "last_updated": datetime.now().isoformat()
                }
            
            # Rate limiting
            await asyncio.sleep(0.1)
        
        return trader_data
    
    async def get_top_traders_from_leaderboard(self, session: aiohttp.ClientSession) -> List[str]:
        """Get top trader addresses from leaderboard"""
        leaderboard = await self.get_leaderboard(session)
        
        # Extract trader addresses from leaderboard
        trader_addresses = []
        for entry in leaderboard[:50]:  # Top 50 traders
            if "user" in entry:
                trader_addresses.append(entry["user"])
        
        return trader_addresses
    
    def filter_profitable_traders(self, trader_data: Dict, min_pnl: float = 1000, min_win_rate: float = 0.55) -> Dict:
        """Filter traders based on profitability criteria"""
        profitable_traders = {}
        
        for address, data in trader_data.items():
            metrics = data["metrics"]
            
            # Filter criteria
            if (metrics["total_pnl"] >= min_pnl and 
                metrics["win_rate"] >= min_win_rate and
                metrics["uses_margin"] and
                metrics["total_trades"] >= 10):  # Minimum trades for reliability
                
                profitable_traders[address] = data
        
        # Sort by total PnL
        sorted_traders = dict(sorted(
            profitable_traders.items(), 
            key=lambda x: x[1]["metrics"]["total_pnl"], 
            reverse=True
        ))
        
        return sorted_traders
    
    def save_to_json(self, data: Dict):
        """Save profitable traders data to JSON file"""
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "total_traders_analyzed": len(data),
            "traders": data
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Saved {len(data)} profitable traders to {self.output_file}")
    
    async def run(self):
        """Main execution function"""
        print("Starting Hyperliquid Profitable Trader Tracker...")
        
        async with aiohttp.ClientSession() as session:
            # Get top traders from leaderboard
            print("Fetching leaderboard data...")
            trader_addresses = await self.get_top_traders_from_leaderboard(session)
            
            if not trader_addresses:
                print("No traders found in leaderboard")
                return
            
            print(f"Found {len(trader_addresses)} traders to analyze")
            
            # Analyze traders
            trader_data = await self.analyze_traders(session, trader_addresses)
            
            # Filter profitable margin traders
            profitable_traders = self.filter_profitable_traders(trader_data)
            
            # Save results
            self.save_to_json(profitable_traders)
            
            # Print summary
            print("\n=== Summary ===")
            print(f"Total traders analyzed: {len(trader_addresses)}")
            print(f"Margin traders found: {len(trader_data)}")
            print(f"Profitable margin traders: {len(profitable_traders)}")
            
            # Display top 5 traders
            print("\n=== Top 5 Profitable Margin Traders ===")
            for i, (address, data) in enumerate(list(profitable_traders.items())[:5]):
                metrics = data["metrics"]
                print(f"\n{i+1}. Trader: {address[:10]}...")
                print(f"   Total PnL: ${metrics['total_pnl']:,.2f}")
                print(f"   Win Rate: {metrics['win_rate']*100:.2f}%")
                print(f"   Max Leverage: {metrics['max_leverage']}x")
                print(f"   Total Trades: {metrics['total_trades']}")

# Configuration class for easy customization
class TrackerConfig:
    def __init__(self):
        self.min_pnl = 1000  # Minimum PnL to be considered profitable
        self.min_win_rate = 0.55  # Minimum win rate (55%)
        self.min_trades = 10  # Minimum number of trades
        self.update_interval = 3600  # Update interval in seconds (1 hour)
        self.max_traders_to_analyze = 50  # Maximum traders to analyze

# Main execution
async def main():
    tracker = HyperliquidTraderTracker()
    
    # Run once
    await tracker.run()
    
    # Optional: Run continuously with updates
    # while True:
    #     await tracker.run()
    #     print(f"\nWaiting {TrackerConfig().update_interval} seconds before next update...")
    #     await asyncio.sleep(TrackerConfig().update_interval)

if __name__ == "__main__":
    # Run the tracker
    asyncio.run(main())