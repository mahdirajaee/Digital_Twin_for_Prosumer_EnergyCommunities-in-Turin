import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from flask import Flask
import io
import base64

from models import Building, Community
from tariffs import ToUTariff, CPPTariff, RTPTariff, EDRTariff
from simulation import CommunitySimulation
from utils import load_electricity_data, parse_uploaded_file
import config

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

app.layout = html.Div([
    html.H1("Prosumer Community Tariff Simulation", className="header"),
    
    dcc.Tabs(id="tabs", value="setup", children=[
        dcc.Tab(label="Setup", value="setup"),
        dcc.Tab(label="Simulation", value="simulation"),
        dcc.Tab(label="Results", value="results"),
        dcc.Tab(label="Analysis", value="analysis")
    ]),
    
    html.Div(id="tab-content")
])

def create_setup_tab():
    return html.Div([
        html.H2("Community Setup"),
        
        html.Div([
            html.H3("Upload Load Profile"),
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
                },
                multiple=False
            ),
            html.Div(id='upload-status'),
        ], className="upload-section"),
        
        html.Div([
            html.H3("Community Parameters"),
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
            
        ], className="parameters-section"),
        
        html.Div([
            html.H3("Tariff Selection"),
            dcc.Dropdown(
                id="tariff-type",
                options=[
                    {"label": "Time-of-Use (ToU)", "value": "tou"},
                    {"label": "Critical Peak Pricing (CPP)", "value": "cpp"},
                    {"label": "Real-Time Pricing (RTP)", "value": "rtp"},
                    {"label": "Emergency Demand Response (EDR)", "value": "edr"}
                ],
                value="tou",
                multi=True
            ),
        ], className="tariff-section"),
        
        html.Button("Initialize Community", id="init-button", className="button-primary"),
        html.Div(id="init-status")
    ])

def create_simulation_tab():
    return html.Div([
        html.H2("Run Simulation"),
        
        html.Div([
            html.Label("Simulation Duration (days):"),
            dcc.Slider(id="sim-duration", min=1, max=365, step=1, value=7,
                      marks={1: "1", 30: "30", 90: "90", 180: "180", 365: "365"}),
            
            html.Label("Price Sensitivity:"),
            dcc.Slider(id="price-sensitivity", min=0.1, max=2.0, step=0.1, value=1.0,
                      marks={i/10: str(i/10) for i in range(1, 21, 5)}),
            
            html.Label("Comfort Threshold:"),
            dcc.Slider(id="comfort-threshold", min=0.5, max=1.0, step=0.05, value=0.8,
                      marks={i/100: str(i/100) for i in range(50, 101, 10)}),
        ], className="simulation-controls"),
        
        html.Button("Run Simulation", id="run-button", className="button-primary"),
        html.Div(id="simulation-progress"),
        
        dcc.Store(id="simulation-results")
    ])

def create_results_tab():
    return html.Div([
        html.H2("Simulation Results"),
        
        html.Div([
            html.H3("Cost Savings Overview"),
            dash_table.DataTable(id="cost-table"),
        ], className="results-section"),
        
        html.Div([
            html.H3("Load Profiles"),
            dcc.Graph(id="load-profile-graph"),
        ], className="results-section"),
        
        html.Div([
            html.H3("Battery Operations"),
            dcc.Graph(id="battery-graph"),
        ], className="results-section"),
        
        html.Div([
            html.H3("Grid Exchange"),
            dcc.Graph(id="grid-graph"),
        ], className="results-section")
    ])

def create_analysis_tab():
    return html.Div([
        html.H2("Performance Analysis"),
        
        html.Div([
            html.H3("Fairness Analysis"),
            dcc.Graph(id="fairness-graph"),
        ], className="analysis-section"),
        
        html.Div([
            html.H3("Peak Reduction"),
            dcc.Graph(id="peak-reduction-graph"),
        ], className="analysis-section"),
        
        html.Div([
            html.H3("Sensitivity Analysis"),
            dcc.Graph(id="sensitivity-graph"),
        ], className="analysis-section")
    ])

@app.callback(Output("tab-content", "children"), [Input("tabs", "value")])
def render_tab_content(active_tab):
    if active_tab == "setup":
        return create_setup_tab()
    elif active_tab == "simulation":
        return create_simulation_tab()
    elif active_tab == "results":
        return create_results_tab()
    elif active_tab == "analysis":
        return create_analysis_tab()

@app.callback(
    Output("upload-status", "children"),
    [Input("upload-data", "contents")],
    [State("upload-data", "filename")]
)
def update_upload_status(contents, filename):
    if contents is not None:
        try:
            df = parse_uploaded_file(contents, filename)
            return html.Div([
                html.P(f"Successfully uploaded {filename}"),
                html.P(f"Data shape: {df.shape[0]} rows, {df.shape[1]} columns"),
                html.P(f"Time range: {df.index[0]} to {df.index[-1]}")
            ], style={"color": "green"})
        except Exception as e:
            return html.Div([
                html.P(f"Error uploading {filename}: {str(e)}")
            ], style={"color": "red"})
    return ""

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
            community = Community(
                num_buildings=num_buildings,
                residents_per_building=residents_per_building,
                pv_capacity=pv_capacity,
                battery_capacity=battery_capacity,
                tariff_types=tariff_types
            )
            return html.Div([
                html.P(f"Community initialized with {num_buildings} buildings"),
                html.P(f"Total residents: {num_buildings * residents_per_building}"),
                html.P(f"Selected tariffs: {', '.join(tariff_types)}")
            ], style={"color": "green"})
        except Exception as e:
            return html.Div([
                html.P(f"Error initializing community: {str(e)}")
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
            sim = CommunitySimulation(community)
            results = sim.run_simulation(
                duration_days=duration,
                price_sensitivity=price_sensitivity,
                comfort_threshold=comfort_threshold
            )
            
            return results, html.Div([
                html.P("Simulation completed successfully!"),
                html.P(f"Duration: {duration} days"),
                html.P(f"Total cost savings: €{results['total_savings']:.2f}")
            ], style={"color": "green"})
        except Exception as e:
            return {}, html.Div([
                html.P(f"Simulation error: {str(e)}")
            ], style={"color": "red"})
    return {}, ""

@app.callback(
    [Output("cost-table", "data"),
     Output("load-profile-graph", "figure"),
     Output("battery-graph", "figure"),
     Output("grid-graph", "figure")],
    [Input("simulation-results", "data")]
)
def update_results(results):
    if not results:
        empty_fig = go.Figure()
        return [], empty_fig, empty_fig, empty_fig
    
    cost_data = results.get('cost_breakdown', [])
    
    load_fig = px.line(
        x=results.get('timestamps', []),
        y=results.get('total_load', []),
        title="Community Load Profile"
    )
    
    battery_fig = px.line(
        x=results.get('timestamps', []),
        y=results.get('battery_soc', []),
        title="Average Battery State of Charge"
    )
    
    grid_fig = px.line(
        x=results.get('timestamps', []),
        y=results.get('grid_exchange', []),
        title="Grid Import/Export"
    )
    
    return cost_data, load_fig, battery_fig, grid_fig

@app.callback(
    [Output("fairness-graph", "figure"),
     Output("peak-reduction-graph", "figure"),
     Output("sensitivity-graph", "figure")],
    [Input("simulation-results", "data")]
)
def update_analysis(results):
    if not results:
        empty_fig = go.Figure()
        return empty_fig, empty_fig, empty_fig
    
    fairness_fig = px.box(
        y=results.get('building_savings', []),
        title="Cost Savings Distribution Across Buildings"
    )
    
    peak_data = results.get('peak_analysis', {})
    peak_fig = go.Figure()
    peak_fig.add_trace(go.Bar(name='Before', x=['Peak Load'], y=[peak_data.get('before', 0)]))
    peak_fig.add_trace(go.Bar(name='After', x=['Peak Load'], y=[peak_data.get('after', 0)]))
    peak_fig.update_layout(title="Peak Load Reduction")
    
    sensitivity_data = results.get('sensitivity_analysis', {})
    sensitivity_fig = px.line(
        x=list(sensitivity_data.keys()),
        y=list(sensitivity_data.values()),
        title="Sensitivity Analysis"
    )
    
    return fairness_fig, peak_fig, sensitivity_fig

if __name__ == "__main__":
    app.run_server(debug=True)