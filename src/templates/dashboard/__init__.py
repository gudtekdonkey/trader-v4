"""
Dashboard Template Components

This module contains modularized components for the portfolio dashboard interface.
The dashboard has been split into reusable components for better maintainability.

Components:
- head.html: Meta tags and external dependencies
- styles.html: Custom CSS styles
- navbar.html: Navigation bar
- overview_metrics.html: Key performance metrics cards
- charts.html: Portfolio P&L and allocation charts
- positions_table.html: Current positions display
- alerts.html: Recent alerts panel
- rebalancing_table.html: Rebalancing recommendations
- dashboard.js: JavaScript functionality

Usage:
    The main dashboard.html template includes these components using Flask's
    template include mechanism. This allows for better code organization and
    the ability to reuse components in other templates if needed.
"""
