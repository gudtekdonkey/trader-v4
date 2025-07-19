// Dashboard JavaScript functions
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

function formatPercentage(value) {
    return (value * 100).toFixed(2) + '%';
}

function loadPortfolioOverview() {
    $.get('/api/portfolio/overview', function(data) {
        if (data.error) {
            console.error(data.error);
            return;
        }

        $('#totalValue').text(formatCurrency(data.total_value));
        $('#totalPnl').text(formatCurrency(data.total_pnl));
        $('#totalReturn').text(data.total_return_pct.toFixed(2) + '%');
        $('#positionCount').text(data.position_count);

        // Update positions table
        const tbody = $('#positionsTable tbody');
        tbody.empty();

        data.positions.forEach(function(pos) {
            const row = `
                <tr>
                    <td><strong>${pos.symbol}</strong></td>
                    <td><span class="badge ${pos.side === 'long' ? 'bg-success' : 'bg-danger'}">${pos.side.toUpperCase()}</span></td>
                    <td>${pos.size.toFixed(4)}</td>
                    <td>${formatCurrency(pos.entry_price)}</td>
                    <td>${formatCurrency(pos.current_price)}</td>
                    <td class="${pos.pnl >= 0 ? 'positive' : 'negative'}">
                        ${formatCurrency(pos.pnl)}<br>
                        <small>(${pos.pnl_pct.toFixed(2)}%)</small>
                    </td>
                    <td>${formatPercentage(pos.weight)}</td>
                </tr>
            `;
            tbody.append(row);
        });
    });
}

// Load all data on page load
$(document).ready(function() {
    loadPortfolioOverview();
    
    // Refresh data every 30 seconds
    setInterval(function() {
        loadPortfolioOverview();
    }, 30000);
});
