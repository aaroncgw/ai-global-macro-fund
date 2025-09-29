"""
Macro Portfolio API Routes

This module provides API endpoints for the global macro ETF trading system,
including portfolio analysis, allocations, and visualizations.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
import os
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import json

# Add the src directory to the path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from src.graph.macro_trading_graph import MacroTradingGraph
from src.config import ETF_UNIVERSE, MACRO_INDICATORS
from src.data_fetchers.macro_fetcher import MacroFetcher

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class PortfolioRequest(BaseModel):
    universe: Optional[List[str]] = None
    date: Optional[str] = "today"
    debug: Optional[bool] = False

class PortfolioResponse(BaseModel):
    universe: List[str]
    final_allocations: Dict[str, float]
    agent_reasoning: Dict[str, Any]
    visualizations: Dict[str, str]
    analysis_summary: Dict[str, Any]

class CorrelationRequest(BaseModel):
    etfs: List[str]
    start_date: Optional[str] = "2020-01-01"
    end_date: Optional[str] = "today"

# Initialize the macro trading graph
macro_graph = None

def get_macro_graph():
    """Get or initialize the macro trading graph."""
    global macro_graph
    if macro_graph is None:
        try:
            macro_graph = MacroTradingGraph(debug=False)
            logger.info("Macro trading graph initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize macro trading graph: {e}")
            raise HTTPException(status_code=500, detail="Failed to initialize macro trading system")
    return macro_graph

@router.post("/analyze", response_model=PortfolioResponse)
async def analyze_macro_portfolio(request: PortfolioRequest):
    """
    Analyze macro portfolio and return allocations with reasoning.
    """
    try:
        # Use provided universe or default
        universe = request.universe or ETF_UNIVERSE[:10]  # Limit to 10 ETFs for performance
        
        logger.info(f"Starting macro portfolio analysis for {len(universe)} ETFs")
        
        # Get macro trading graph
        graph = get_macro_graph()
        
        # Run the complete workflow
        complete_state = graph.propagate(universe, request.date)
        
        # Extract results
        final_allocations = complete_state.get('final_allocations', {})
        agent_reasoning = complete_state.get('agent_reasoning', {})
        etf_data = complete_state.get('etf_data', pd.DataFrame())
        
        # Generate visualizations
        visualizations = await generate_visualizations(etf_data, universe, final_allocations)
        
        # Generate analysis summary
        analysis_summary = generate_analysis_summary(complete_state, final_allocations)
        
        logger.info(f"Macro portfolio analysis completed successfully")
        
        return PortfolioResponse(
            universe=universe,
            final_allocations=final_allocations,
            agent_reasoning=agent_reasoning,
            visualizations=visualizations,
            analysis_summary=analysis_summary
        )
        
    except Exception as e:
        logger.error(f"Macro portfolio analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/correlation")
async def get_correlation_matrix(request: CorrelationRequest):
    """
    Generate ETF correlation matrix and heatmap visualization.
    """
    try:
        # Initialize data fetcher
        fetcher = MacroFetcher()
        
        # Fetch ETF data
        etf_data = fetcher.fetch_etf_data(request.etfs, request.start_date, request.end_date)
        
        if etf_data.empty:
            raise HTTPException(status_code=400, detail="No ETF data available")
        
        # Calculate correlation matrix
        returns = etf_data.pct_change().dropna()
        corr_matrix = returns.corr()
        
        # Create correlation heatmap
        fig = px.imshow(
            corr_matrix,
            x=request.etfs,
            y=request.etfs,
            title='ETF Correlation Heatmap',
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        
        # Convert to JSON
        correlation_plot = json.dumps(fig, cls=PlotlyJSONEncoder)
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "visualization": correlation_plot,
            "etfs": request.etfs,
            "period": f"{request.start_date} to {request.end_date}"
        }
        
    except Exception as e:
        logger.error(f"Correlation analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Correlation analysis failed: {str(e)}")

@router.get("/dashboard", response_class=HTMLResponse)
async def macro_portfolio_dashboard():
    """
    Serve the macro portfolio dashboard HTML page.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Macro Portfolio Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                color: #333;
            }
            .controls {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
                flex-wrap: wrap;
            }
            .control-group {
                display: flex;
                flex-direction: column;
                gap: 5px;
            }
            .control-group label {
                font-weight: bold;
                color: #555;
            }
            .control-group input, .control-group select {
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            button {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #0056b3;
            }
            button:disabled {
                background-color: #ccc;
                cursor: not-allowed;
            }
            .results {
                margin-top: 20px;
            }
            .allocation-table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            .allocation-table th, .allocation-table td {
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }
            .allocation-table th {
                background-color: #f8f9fa;
                font-weight: bold;
            }
            .allocation-table tr:nth-child(even) {
                background-color: #f8f9fa;
            }
            .chart-container {
                margin: 20px 0;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px;
            }
            .loading {
                text-align: center;
                padding: 20px;
                color: #666;
            }
            .error {
                color: #dc3545;
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                padding: 10px;
                border-radius: 4px;
                margin: 10px 0;
            }
            .success {
                color: #155724;
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                padding: 10px;
                border-radius: 4px;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌍 Global Macro ETF Trading System</h1>
                <p>AI-Powered Portfolio Analysis & Allocation</p>
            </div>
            
            <div class="controls">
                <div class="control-group">
                    <label for="etfUniverse">ETF Universe:</label>
                    <input type="text" id="etfUniverse" placeholder="SPY,QQQ,TLT,GLD" value="SPY,QQQ,TLT,GLD">
                </div>
                <div class="control-group">
                    <label for="analysisDate">Analysis Date:</label>
                    <input type="date" id="analysisDate">
                </div>
                <div class="control-group">
                    <label for="debugMode">Debug Mode:</label>
                    <select id="debugMode">
                        <option value="false">False</option>
                        <option value="true">True</option>
                    </select>
                </div>
                <div class="control-group">
                    <button onclick="analyzePortfolio()" id="analyzeBtn">Analyze Portfolio</button>
                </div>
            </div>
            
            <div id="results" class="results" style="display: none;">
                <div id="loading" class="loading" style="display: none;">
                    <h3>🔄 Analyzing Portfolio...</h3>
                    <p>This may take a few minutes as we analyze macro data, geopolitical events, and optimize allocations.</p>
                </div>
                
                <div id="error" class="error" style="display: none;"></div>
                
                <div id="success" class="success" style="display: none;"></div>
                
                <div id="allocations"></div>
                <div id="visualizations"></div>
            </div>
        </div>
        
        <script>
            // Set default date to today
            document.getElementById('analysisDate').value = new Date().toISOString().split('T')[0];
            
            async function analyzePortfolio() {
                const etfUniverse = document.getElementById('etfUniverse').value.split(',').map(etf => etf.trim());
                const analysisDate = document.getElementById('analysisDate').value;
                const debugMode = document.getElementById('debugMode').value === 'true';
                
                // Show loading
                document.getElementById('results').style.display = 'block';
                document.getElementById('loading').style.display = 'block';
                document.getElementById('error').style.display = 'none';
                document.getElementById('success').style.display = 'none';
                document.getElementById('analyzeBtn').disabled = true;
                
                try {
                    const response = await fetch('/api/macro-portfolio/analyze', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            universe: etfUniverse,
                            date: analysisDate,
                            debug: debugMode
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    
                    // Hide loading
                    document.getElementById('loading').style.display = 'none';
                    
                    // Show success message
                    document.getElementById('success').style.display = 'block';
                    document.getElementById('success').innerHTML = `
                        <h3>✅ Analysis Complete!</h3>
                        <p>Successfully analyzed ${data.universe.length} ETFs with comprehensive reasoning.</p>
                    `;
                    
                    // Display allocations
                    displayAllocations(data.final_allocations);
                    
                    // Display visualizations
                    displayVisualizations(data.visualizations);
                    
                } catch (error) {
                    console.error('Error:', error);
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('error').style.display = 'block';
                    document.getElementById('error').innerHTML = `
                        <h3>❌ Analysis Failed</h3>
                        <p>Error: ${error.message}</p>
                    `;
                } finally {
                    document.getElementById('analyzeBtn').disabled = false;
                }
            }
            
            function displayAllocations(allocations) {
                const allocationsDiv = document.getElementById('allocations');
                
                // Sort allocations by percentage
                const sortedAllocations = Object.entries(allocations)
                    .sort(([,a], [,b]) => b - a);
                
                let html = '<h3>📊 Final Portfolio Allocations</h3>';
                html += '<table class="allocation-table">';
                html += '<thead><tr><th>Rank</th><th>ETF</th><th>Allocation</th><th>Percentage</th></tr></thead>';
                html += '<tbody>';
                
                sortedAllocations.forEach(([etf, allocation], index) => {
                    html += `<tr>
                        <td>${index + 1}</td>
                        <td><strong>${etf}</strong></td>
                        <td>${allocation.toFixed(1)}%</td>
                        <td>
                            <div style="background-color: #e3f2fd; height: 20px; border-radius: 10px; position: relative;">
                                <div style="background-color: #2196f3; height: 100%; width: ${allocation}%; border-radius: 10px;"></div>
                                <span style="position: absolute; top: 2px; left: 50%; transform: translateX(-50%); font-size: 12px; color: #333;">${allocation.toFixed(1)}%</span>
                            </div>
                        </td>
                    </tr>`;
                });
                
                html += '</tbody></table>';
                allocationsDiv.innerHTML = html;
            }
            
            function displayVisualizations(visualizations) {
                const visualizationsDiv = document.getElementById('visualizations');
                
                let html = '<h3>📈 Visualizations</h3>';
                
                // Display correlation heatmap if available
                if (visualizations.correlation_heatmap) {
                    html += '<div class="chart-container">';
                    html += '<h4>ETF Correlation Heatmap</h4>';
                    html += '<div id="correlation-chart"></div>';
                    html += '</div>';
                    
                    // Render correlation chart
                    const correlationData = JSON.parse(visualizations.correlation_heatmap);
                    Plotly.newPlot('correlation-chart', correlationData.data, correlationData.layout);
                }
                
                // Display allocation pie chart if available
                if (visualizations.allocation_pie) {
                    html += '<div class="chart-container">';
                    html += '<h4>Portfolio Allocation</h4>';
                    html += '<div id="allocation-chart"></div>';
                    html += '</div>';
                    
                    // Render allocation chart
                    const allocationData = JSON.parse(visualizations.allocation_pie);
                    Plotly.newPlot('allocation-chart', allocationData.data, allocationData.layout);
                }
                
                visualizationsDiv.innerHTML = html;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

async def generate_visualizations(etf_data: pd.DataFrame, universe: List[str], allocations: Dict[str, float]) -> Dict[str, str]:
    """Generate visualizations for the portfolio analysis."""
    visualizations = {}
    
    try:
        # Correlation heatmap
        if not etf_data.empty and len(universe) > 1:
            returns = etf_data.pct_change().dropna()
            if not returns.empty:
                corr_matrix = returns.corr()
                
                fig = px.imshow(
                    corr_matrix,
                    x=universe,
                    y=universe,
                    title='ETF Correlation Heatmap',
                    color_continuous_scale='RdBu',
                    aspect='auto'
                )
                fig.update_layout(
                    width=600,
                    height=500,
                    title_x=0.5
                )
                
                visualizations['correlation_heatmap'] = json.dumps(fig, cls=PlotlyJSONEncoder)
        
        # Allocation pie chart
        if allocations:
            # Filter out zero allocations
            filtered_allocations = {k: v for k, v in allocations.items() if v > 0}
            
            if filtered_allocations:
                fig = go.Figure(data=[go.Pie(
                    labels=list(filtered_allocations.keys()),
                    values=list(filtered_allocations.values()),
                    hole=0.3
                )])
                
                fig.update_layout(
                    title="Portfolio Allocation",
                    width=500,
                    height=400,
                    title_x=0.5
                )
                
                visualizations['allocation_pie'] = json.dumps(fig, cls=PlotlyJSONEncoder)
    
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}")
    
    return visualizations

def generate_analysis_summary(complete_state: Dict[str, Any], allocations: Dict[str, float]) -> Dict[str, Any]:
    """Generate a summary of the analysis results."""
    summary = {
        "total_etfs": len(complete_state.get('universe', [])),
        "analysis_date": complete_state.get('date', 'unknown'),
        "total_allocation": sum(allocations.values()) if allocations else 0,
        "top_allocation": max(allocations.items(), key=lambda x: x[1]) if allocations else None,
        "agent_count": len(complete_state.get('agent_reasoning', {})),
        "has_macro_data": bool(complete_state.get('macro_data')),
        "has_news_data": bool(complete_state.get('news')),
        "has_etf_data": not complete_state.get('etf_data', pd.DataFrame()).empty
    }
    
    return summary
