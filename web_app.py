#!/usr/bin/env python3

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from flask import Flask
import json

from run_simulation import SimpleCommunity, SimpleCommunitySimulation
import config_file as config

server = Flask(__name__)
app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("Prosumer Community Tariff Simulation - Turin, Italy", style={'textAlign': 'center', 'color': '#2c3e50'}),
    html.P("Community energy simulation with EUR pricing and Turin continental climate", 
           style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '20px'}),
    
    dcc.Tabs(id="tabs", value="setup", children=[
        dcc.Tab(label="Setup", value="setup"),
        dcc.Tab(label="Simulation", value="simulation"),
        dcc.Tab(label="Results", value="results"),
    ]),
    
    html.Div(id="tab-content"),
    dcc.Store(id="simulation-results")
])

def create_setup_tab():
    return html.Div([
        html.H2("Community Setup"),
        
        html.Div([
            html.Label("Number of Buildings:"),
            dcc.Slider(id="num-buildings", min=10, max=100, step=10, value=50, 
                      marks={i: str(i) for i in range(10, 101, 20)}),
            
            html.Label("Residents per Building:"),
            dcc.Slider(id="residents-per-building", min=10, max=50, step=10, value=20,
                      marks={i: str(i) for i in range(10, 51, 10)}),
            
            html.Label("PV Capacity (kW):"),
            dcc.Slider(id="pv-capacity", min=5, max=20, step=1, value=10,
                      marks={i: str(i) for i in range(5, 21, 5)}),
            
            html.Label("Battery Capacity (kWh):"),
            dcc.Slider(id="battery-capacity", min=10, max=50, step=5, value=25,
                      marks={i: str(i) for i in range(10, 51, 10)}),
            
        ], style={'padding': '20px', 'margin': '20px', 'border': '1px solid #ddd'}),
        
        html.Div([
            html.Label("Tariff Selection:"),
            dcc.Dropdown(
                id="tariff-type",
                options=[
                    {"label": "Time-of-Use (ToU)", "value": "tou"},
                    {"label": "Critical Peak Pricing (CPP)", "value": "cpp"},
                    {"label": "Real-Time Pricing (RTP)", "value": "rtp"},
                    {"label": "Emergency Demand Response (EDR)", "value": "edr"}
                ],
                value=["tou", "cpp"],
                multi=True
            ),
        ], style={'padding': '20px', 'margin': '20px', 'border': '1px solid #ddd'}),
        
        html.Button("Initialize Community", id="init-button", 
                   style={'backgroundColor': '#3498db', 'color': 'white', 'padding': '10px 20px', 
                         'border': 'none', 'borderRadius': '5px', 'margin': '20px'}),
        html.Div(id="init-status")
    ])

def create_simulation_tab():
    return html.Div([
        html.H2("Run Simulation"),
        
        html.Div([
            html.Label("Simulation Duration (days):"),
            dcc.Slider(id="sim-duration", min=1, max=30, step=1, value=7,
                      marks={1: "1", 7: "7", 14: "14", 30: "30"}),
            
            html.Label("Price Sensitivity:"),
            dcc.Slider(id="price-sensitivity", min=0.1, max=2.0, step=0.1, value=1.0,
                      marks={i/10: str(i/10) for i in range(1, 21, 5)}),
            
            html.Label("Comfort Threshold:"),
            dcc.Slider(id="comfort-threshold", min=0.5, max=1.0, step=0.05, value=0.8,
                      marks={i/100: str(i/100) for i in range(50, 101, 10)}),
        ], style={'padding': '20px', 'margin': '20px', 'border': '1px solid #ddd'}),
        
        html.Button("Run Simulation", id="run-button", 
                   style={'backgroundColor': '#27ae60', 'color': 'white', 'padding': '10px 20px', 
                         'border': 'none', 'borderRadius': '5px', 'margin': '20px'}),
        html.Div(id="simulation-progress"),
    ])

def create_results_tab():
    return html.Div([
        html.H2("Simulation Results"),
        
        html.Div([
            html.H3("Financial Summary"),
            html.Div(id="financial-summary"),
        ], style={'padding': '20px', 'margin': '20px', 'border': '1px solid #ddd'}),
        
        html.Div([
            html.H3("Load Profiles"),
            dcc.Graph(id="load-profile-graph"),
        ], style={'padding': '20px', 'margin': '20px', 'border': '1px solid #ddd'}),
        
        html.Div([
            html.H3("Grid Exchange"),
            dcc.Graph(id="grid-graph"),
        ], style={'padding': '20px', 'margin': '20px', 'border': '1px solid #ddd'}),
        
        html.Div([
            html.H3("Tariff Comparison"),
            dcc.Graph(id="tariff-comparison-graph"),
        ], style={'padding': '20px', 'margin': '20px', 'border': '1px solid #ddd'}),
    ])

@app.callback(Output("tab-content", "children"), [Input("tabs", "value")])
def render_tab_content(active_tab):
    if active_tab == "setup":
        return create_setup_tab()
    elif active_tab == "simulation":
        return create_simulation_tab()
    elif active_tab == "results":
        return create_results_tab()

@app.callback(
    Output("init-status", "children"),
    [Input("init-button", "n_clicks")],
    [State("num-buildings", "value"),
     State("residents-per-building", "value"),
     State("pv-capacity", "value"),
     State("battery-capacity", "value"),
     State("tariff-type", "value")]
)
def initialize_community(n_clicks, num_buildings, residents_per_building, 
                        pv_capacity, battery_capacity, tariff_types):
    if n_clicks:
        try:
            global community
            community = SimpleCommunity(
                num_buildings=num_buildings,
                residents_per_building=residents_per_building,
                pv_capacity=pv_capacity,
                battery_capacity=battery_capacity,
                tariff_types=tariff_types or ['tou']
            )
            return html.Div([
                html.P(f"✅ Community initialized with {num_buildings} buildings"),
                html.P(f"Total residents: {num_buildings * residents_per_building}"),
                html.P(f"Selected tariffs: {', '.join(tariff_types or ['tou'])}")
            ], style={"color": "green"})
        except Exception as e:
            return html.Div([
                html.P(f"❌ Error initializing community: {str(e)}")
            ], style={"color": "red"})
    return ""

@app.callback(
    [Output("simulation-results", "data"),
     Output("simulation-progress", "children")],
    [Input("run-button", "n_clicks")],
    [State("sim-duration", "value"),
     State("price-sensitivity", "value"),
     State("comfort-threshold", "value")]
)
def run_simulation(n_clicks, duration, price_sensitivity, comfort_threshold):
    if n_clicks and 'community' in globals():
        try:
            sim = SimpleCommunitySimulation(community)
            results = sim.run_simulation(
                duration_days=duration,
                price_sensitivity=price_sensitivity,
                comfort_threshold=comfort_threshold
            )
            
            return results, html.Div([
                html.P("✅ Simulation completed successfully!"),
                html.P(f"Duration: {duration} days"),
                html.P(f"Best tariff: {results['simulation_summary']['best_tariff']}"),
                html.P(f"Total savings: ${results['simulation_summary']['total_savings']:.2f}")
            ], style={"color": "green"})
        except Exception as e:
            return {}, html.Div([
                html.P(f"❌ Simulation error: {str(e)}")
            ], style={"color": "red"})
    return {}, html.Div([
        html.P("⚠️ Please initialize community first in the Setup tab")
    ], style={"color": "orange"})

@app.callback(
    Output("financial-summary", "children"),
    [Input("simulation-results", "data")]
)
def update_financial_summary(results):
    if not results:
        return "No simulation results available"
    
    summary = results['simulation_summary']
    fairness = results['fairness_analysis']
    
    return html.Div([
        html.Div([
            html.H4(f"Best Tariff: {summary['best_tariff'].upper()}"),
            html.P(f"Baseline Cost: €{summary['baseline_cost']:.2f}"),
            html.P(f"Optimized Cost: €{summary['optimized_cost']:.2f}"),
            html.P(f"Total Savings: ${summary['total_savings']:.2f} ({summary['savings_percent']:.1f}%)"),
        ], style={'display': 'inline-block', 'width': '48%', 'verticalAlign': 'top'}),
        
        html.Div([
            html.H4("Fairness Metrics"),
            html.P(f"Gini Coefficient: {fairness['gini_coefficient']:.3f}"),
            html.P(f"Coefficient of Variation: {fairness['coefficient_of_variation']:.3f}"),
            html.P(f"Cost Range: €{fairness['cost_range']:.2f}"),
        ], style={'display': 'inline-block', 'width': '48%', 'verticalAlign': 'top', 'marginLeft': '4%'})
    ])

@app.callback(
    Output("load-profile-graph", "figure"),
    [Input("simulation-results", "data")]
)
def update_load_profile(results):
    if not results:
        return go.Figure()
    
    timestamps = pd.to_datetime(results['timestamps'])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=results['total_load'],
        name='Total Load', line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=results['pv_generation'],
        name='PV Generation', line=dict(color='orange', width=2)
    ))
    
    fig.update_layout(
        title="Community Load Profile vs PV Generation",
        xaxis_title="Time",
        yaxis_title="Power (kW)",
        hovermode='x unified'
    )
    
    return fig

@app.callback(
    Output("grid-graph", "figure"),
    [Input("simulation-results", "data")]
)
def update_grid_graph(results):
    if not results:
        return go.Figure()
    
    timestamps = pd.to_datetime(results['timestamps'])
    grid_exchange = np.array(results['grid_exchange'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps, y=grid_exchange,
        fill='tozeroy', name='Grid Exchange',
        line=dict(color='red', width=2)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    
    fig.update_layout(
        title="Grid Import/Export",
        xaxis_title="Time",
        yaxis_title="Power (kW)",
        annotations=[
            dict(x=0.02, y=0.98, xref="paper", yref="paper",
                 text="Positive: Import from Grid<br>Negative: Export to Grid",
                 showarrow=False, bgcolor="white", bordercolor="black")
        ]
    )
    
    return fig

@app.callback(
    Output("tariff-comparison-graph", "figure"),
    [Input("simulation-results", "data")]
)
def update_tariff_comparison(results):
    if not results:
        return go.Figure()
    
    tariff_comparison = results['tariff_comparison']
    
    tariffs = list(tariff_comparison.keys())
    costs = [tariff_comparison[t]['total_cost'] for t in tariffs]
    savings_pct = [tariff_comparison[t]['cost_savings_percent'] for t in tariffs]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=tariffs, y=costs,
        name='Total Cost (€)',
        marker_color='lightblue',
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=tariffs, y=savings_pct,
        name='Savings (%)',
        mode='markers+lines',
        marker=dict(size=10, color='red'),
        line=dict(color='red', width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Tariff Performance Comparison",
        xaxis_title="Tariff Type",
        yaxis=dict(title="Total Cost (€)", side='left'),
        yaxis2=dict(title="Savings (%)", side='right', overlaying='y'),
        hovermode='x unified'
    )
    
    return fig

if __name__ == "__main__":
    print("Starting Prosumer Community Simulation Web App...")
    print("Open your browser and go to: http://127.0.0.1:8050")
    app.run_server(debug=True, port=8050)
