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

# Modern CSS styling
external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
]

app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)

# Define modern color scheme and styles
COLORS = {
    'primary': '#2E86C1',
    'secondary': '#F39C12',
    'success': '#27AE60',
    'warning': '#F39C12',
    'danger': '#E74C3C',
    'info': '#17A2B8',
    'light': '#F8F9FA',
    'dark': '#2C3E50',
    'muted': '#6C757D',
    'background': '#F5F7FA',
    'card': '#FFFFFF',
    'border': '#E9ECEF'
}

# Common styles
CARD_STYLE = {
    'backgroundColor': COLORS['card'],
    'borderRadius': '12px',
    'boxShadow': '0 2px 20px rgba(0,0,0,0.08)',
    'padding': '24px',
    'margin': '16px 0',
    'border': 'none',
    'className': 'dash-card'
}

BUTTON_STYLE = {
    'backgroundColor': COLORS['primary'],
    'color': 'white',
    'padding': '12px 24px',
    'border': 'none',
    'borderRadius': '8px',
    'fontSize': '14px',
    'fontWeight': '500',
    'cursor': 'pointer',
    'transition': 'all 0.2s ease',
    'margin': '8px',
    'className': 'dash-button'
}

SUCCESS_BUTTON_STYLE = {
    **BUTTON_STYLE,
    'backgroundColor': COLORS['success'],
    'className': 'dash-button success-button'
}

TAB_STYLE = {
    'backgroundColor': COLORS['light'],
    'border': 'none',
    'borderRadius': '8px 8px 0 0',
    'padding': '12px 24px',
    'fontWeight': '500',
    'color': COLORS['muted']
}

TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    'backgroundColor': COLORS['primary'],
    'color': 'white'
}

app.layout = html.Div([
    # Header Section
    html.Div([
        html.Div([
            html.Div([
                html.I(className="fas fa-bolt", style={'fontSize': '28px', 'marginRight': '12px', 'color': COLORS['secondary']}),
                html.H1("TurinEnergyTwin", style={
                    'fontSize': '32px', 
                    'fontWeight': '700', 
                    'margin': '0',
                    'color': COLORS['dark'],
                    'display': 'inline-block'
                }),
            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
            
            html.P("Digital Twin Framework for Prosumer Community Tariff Optimization", 
                   style={
                       'textAlign': 'center', 
                       'color': COLORS['muted'], 
                       'fontSize': '16px',
                       'margin': '8px 0 0 0',
                       'fontWeight': '400'
                   }),
            
            html.Div([
                html.Span([
                    html.I(className="fas fa-map-marker-alt", style={'marginRight': '6px'}),
                    "Turin, Italy"
                ], style={'marginRight': '20px', 'color': COLORS['muted']}),
                html.Span([
                    html.I(className="fas fa-euro-sign", style={'marginRight': '6px'}),
                    "EUR Pricing"
                ], style={'marginRight': '20px', 'color': COLORS['muted']}),
                html.Span([
                    html.I(className="fas fa-thermometer-half", style={'marginRight': '6px'}),
                    "Continental Climate"
                ], style={'color': COLORS['muted']}),
            ], style={
                'textAlign': 'center', 
                'marginTop': '12px',
                'fontSize': '14px'
            })
        ], style={
            'backgroundColor': COLORS['card'],
            'padding': '32px',
            'borderRadius': '16px',
            'boxShadow': '0 4px 30px rgba(0,0,0,0.1)',
            'margin': '20px',
            'border': f'1px solid {COLORS["border"]}'
        })
    ], style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'fontFamily': 'Inter, sans-serif'}),
    
    # Navigation Tabs
    html.Div([
        dcc.Tabs(id="tabs", value="setup", children=[
            dcc.Tab(
                label="🏗️ Setup", 
                value="setup",
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE
            ),
            dcc.Tab(
                label="⚡ Simulation", 
                value="simulation",
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE
            ),
            dcc.Tab(
                label="📊 Results", 
                value="results",
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE
            ),
        ], style={
            'margin': '0 20px',
            'borderBottom': 'none'
        })
    ]),
    
    # Content Area
    html.Div([
        html.Div(id="tab-content")
    ], style={'margin': '0 20px 20px 20px'}),
    
    # Footer
    html.Div([
        html.Div([
            html.Div([
                html.I(className="fas fa-university", style={'marginRight': '8px', 'color': COLORS['primary']}),
                html.Span("TurinEnergyTwin", style={'fontWeight': '600', 'color': COLORS['dark']})
            ], style={'marginBottom': '8px'}),
            
            html.P("Digital Twin Framework for Prosumer Community Tariff Optimization", 
                   style={'color': COLORS['muted'], 'fontSize': '14px', 'margin': '0 0 8px 0'}),
            
            html.Div([
                html.Span([
                    html.I(className="fas fa-map-marker-alt", style={'marginRight': '4px'}),
                    "Turin, Italy"
                ], style={'marginRight': '16px', 'fontSize': '12px', 'color': COLORS['muted']}),
                html.Span([
                    html.I(className="fas fa-code", style={'marginRight': '4px'}),
                    "Powered by Dash & Plotly"
                ], style={'marginRight': '16px', 'fontSize': '12px', 'color': COLORS['muted']}),
                html.Span([
                    html.I(className="fas fa-calendar", style={'marginRight': '4px'}),
                    "2025"
                ], style={'fontSize': '12px', 'color': COLORS['muted']})
            ])
        ], style={
            'textAlign': 'center',
            'padding': '24px',
            'backgroundColor': COLORS['card'],
            'borderRadius': '12px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.05)',
            'margin': '20px',
            'border': f'1px solid {COLORS["border"]}'
        })
    ], style={'backgroundColor': COLORS['background'], 'paddingBottom': '20px'}),
    
    # Data Store
    dcc.Store(id="simulation-results")
], style={
    'backgroundColor': COLORS['background'],
    'minHeight': '100vh',
    'fontFamily': 'Inter, sans-serif'
})

def create_setup_tab():
    return html.Div([
        # Page Header
        html.Div([
            html.H2([
                html.I(className="fas fa-cogs", style={'marginRight': '12px', 'color': COLORS['primary']}),
                "Community Configuration"
            ], style={'color': COLORS['dark'], 'fontWeight': '600', 'margin': '0 0 8px 0'}),
            html.P("Configure your prosumer community parameters for the simulation", 
                   style={'color': COLORS['muted'], 'margin': '0', 'fontSize': '16px'})
        ], style=CARD_STYLE),
        
        # Community Parameters Section
        html.Div([
            html.H3([
                html.I(className="fas fa-building", style={'marginRight': '8px', 'color': COLORS['secondary']}),
                "Community Parameters"
            ], style={'color': COLORS['dark'], 'fontWeight': '600', 'marginBottom': '24px'}),
            
            # Grid Layout for Parameters
            html.Div([
                # Buildings Parameter
                html.Div([
                    html.Label([
                        html.I(className="fas fa-home", style={'marginRight': '8px', 'color': COLORS['primary']}),
                        "Number of Buildings"
                    ], style={'fontSize': '14px', 'fontWeight': '500', 'color': COLORS['dark'], 'marginBottom': '8px', 'display': 'block'}),
                    dcc.Slider(
                        id="num-buildings", 
                        min=10, max=100, step=10, value=50,
                        marks={i: {'label': str(i), 'style': {'color': COLORS['muted'], 'fontSize': '12px'}} 
                               for i in range(10, 101, 20)},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    html.Div(id="buildings-value", style={'textAlign': 'center', 'marginTop': '8px', 'color': COLORS['primary'], 'fontWeight': '500'})
                ], style={'marginBottom': '24px'}),
                
                # Residents Parameter
                html.Div([
                    html.Label([
                        html.I(className="fas fa-users", style={'marginRight': '8px', 'color': COLORS['primary']}),
                        "Residents per Building"
                    ], style={'fontSize': '14px', 'fontWeight': '500', 'color': COLORS['dark'], 'marginBottom': '8px', 'display': 'block'}),
                    dcc.Slider(
                        id="residents-per-building", 
                        min=10, max=50, step=10, value=20,
                        marks={i: {'label': str(i), 'style': {'color': COLORS['muted'], 'fontSize': '12px'}} 
                               for i in range(10, 51, 10)},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    html.Div(id="residents-value", style={'textAlign': 'center', 'marginTop': '8px', 'color': COLORS['primary'], 'fontWeight': '500'})
                ], style={'marginBottom': '24px'}),
                
                # PV Capacity Parameter
                html.Div([
                    html.Label([
                        html.I(className="fas fa-solar-panel", style={'marginRight': '8px', 'color': COLORS['secondary']}),
                        "PV Capacity (kW)"
                    ], style={'fontSize': '14px', 'fontWeight': '500', 'color': COLORS['dark'], 'marginBottom': '8px', 'display': 'block'}),
                    dcc.Slider(
                        id="pv-capacity", 
                        min=5, max=20, step=1, value=10,
                        marks={i: {'label': str(i), 'style': {'color': COLORS['muted'], 'fontSize': '12px'}} 
                               for i in range(5, 21, 5)},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    html.Div(id="pv-value", style={'textAlign': 'center', 'marginTop': '8px', 'color': COLORS['secondary'], 'fontWeight': '500'})
                ], style={'marginBottom': '24px'}),
                
                # Battery Capacity Parameter
                html.Div([
                    html.Label([
                        html.I(className="fas fa-battery-three-quarters", style={'marginRight': '8px', 'color': COLORS['success']}),
                        "Battery Capacity (kWh)"
                    ], style={'fontSize': '14px', 'fontWeight': '500', 'color': COLORS['dark'], 'marginBottom': '8px', 'display': 'block'}),
                    dcc.Slider(
                        id="battery-capacity", 
                        min=10, max=50, step=5, value=25,
                        marks={i: {'label': str(i), 'style': {'color': COLORS['muted'], 'fontSize': '12px'}} 
                               for i in range(10, 51, 10)},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    html.Div(id="battery-value", style={'textAlign': 'center', 'marginTop': '8px', 'color': COLORS['success'], 'fontWeight': '500'})
                ], style={'marginBottom': '24px'})
            ])
        ], style=CARD_STYLE),
        
        # Tariff Selection Section
        html.Div([
            html.H3([
                html.I(className="fas fa-euro-sign", style={'marginRight': '8px', 'color': COLORS['warning']}),
                "Tariff Configuration"
            ], style={'color': COLORS['dark'], 'fontWeight': '600', 'marginBottom': '24px'}),
            
            html.Label([
                html.I(className="fas fa-list-ul", style={'marginRight': '8px', 'color': COLORS['primary']}),
                "Select Tariff Types to Compare"
            ], style={'fontSize': '14px', 'fontWeight': '500', 'color': COLORS['dark'], 'marginBottom': '12px', 'display': 'block'}),
            
            dcc.Dropdown(
                id="tariff-type",
                options=[
                    {"label": "⏰ Time-of-Use (ToU)", "value": "tou"},
                    {"label": "🔥 Critical Peak Pricing (CPP)", "value": "cpp"},
                    {"label": "📈 Real-Time Pricing (RTP)", "value": "rtp"},
                    {"label": "🚨 Emergency Demand Response (EDR)", "value": "edr"}
                ],
                value=["tou", "cpp"],
                multi=True,
                style={'fontSize': '14px'},
                placeholder="Select tariff types to analyze..."
            ),
            
            html.Div([
                html.P("💡 Tip: Select multiple tariffs to compare their performance", 
                       style={'color': COLORS['info'], 'fontSize': '13px', 'margin': '12px 0 0 0', 'fontStyle': 'italic'})
            ])
        ], style=CARD_STYLE),
        
        # Action Section
        html.Div([
            html.Button([
                html.I(className="fas fa-rocket", style={'marginRight': '8px'}),
                "Initialize Community"
            ], id="init-button", style=SUCCESS_BUTTON_STYLE),
            
            html.Div(id="init-status", style={'marginTop': '16px'})
        ], style={'textAlign': 'center', 'margin': '24px 0'})
    ])

def create_simulation_tab():
    return html.Div([
        # Page Header
        html.Div([
            html.H2([
                html.I(className="fas fa-play-circle", style={'marginRight': '12px', 'color': COLORS['success']}),
                "Simulation Control"
            ], style={'color': COLORS['dark'], 'fontWeight': '600', 'margin': '0 0 8px 0'}),
            html.P("Configure simulation parameters and run the optimization", 
                   style={'color': COLORS['muted'], 'margin': '0', 'fontSize': '16px'})
        ], style=CARD_STYLE),
        
        # Simulation Parameters
        html.Div([
            html.H3([
                html.I(className="fas fa-sliders-h", style={'marginRight': '8px', 'color': COLORS['primary']}),
                "Simulation Parameters"
            ], style={'color': COLORS['dark'], 'fontWeight': '600', 'marginBottom': '24px'}),
            
            # Grid Layout for Parameters
            html.Div([
                # Duration Parameter
                html.Div([
                    html.Label([
                        html.I(className="fas fa-calendar-alt", style={'marginRight': '8px', 'color': COLORS['primary']}),
                        "Simulation Duration (days)"
                    ], style={'fontSize': '14px', 'fontWeight': '500', 'color': COLORS['dark'], 'marginBottom': '8px', 'display': 'block'}),
                    dcc.Slider(
                        id="sim-duration", 
                        min=1, max=30, step=1, value=7,
                        marks={1: {'label': '1 day', 'style': {'color': COLORS['muted'], 'fontSize': '12px'}}, 
                               7: {'label': '1 week', 'style': {'color': COLORS['muted'], 'fontSize': '12px'}}, 
                               14: {'label': '2 weeks', 'style': {'color': COLORS['muted'], 'fontSize': '12px'}}, 
                               30: {'label': '1 month', 'style': {'color': COLORS['muted'], 'fontSize': '12px'}}},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    html.Div(id="duration-value", style={'textAlign': 'center', 'marginTop': '8px', 'color': COLORS['primary'], 'fontWeight': '500'})
                ], style={'marginBottom': '32px'}),
                
                # Price Sensitivity Parameter
                html.Div([
                    html.Label([
                        html.I(className="fas fa-chart-line", style={'marginRight': '8px', 'color': COLORS['warning']}),
                        "Price Sensitivity"
                    ], style={'fontSize': '14px', 'fontWeight': '500', 'color': COLORS['dark'], 'marginBottom': '8px', 'display': 'block'}),
                    dcc.Slider(
                        id="price-sensitivity", 
                        min=0.1, max=2.0, step=0.1, value=1.0,
                        marks={0.5: {'label': 'Low', 'style': {'color': COLORS['muted'], 'fontSize': '12px'}}, 
                               1.0: {'label': 'Medium', 'style': {'color': COLORS['muted'], 'fontSize': '12px'}}, 
                               1.5: {'label': 'High', 'style': {'color': COLORS['muted'], 'fontSize': '12px'}}, 
                               2.0: {'label': 'Very High', 'style': {'color': COLORS['muted'], 'fontSize': '12px'}}},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    html.Div(id="sensitivity-value", style={'textAlign': 'center', 'marginTop': '8px', 'color': COLORS['warning'], 'fontWeight': '500'}),
                    html.P("How responsive residents are to electricity price changes", 
                           style={'color': COLORS['muted'], 'fontSize': '12px', 'textAlign': 'center', 'marginTop': '8px', 'fontStyle': 'italic'})
                ], style={'marginBottom': '32px'}),
                
                # Comfort Threshold Parameter
                html.Div([
                    html.Label([
                        html.I(className="fas fa-thermometer-half", style={'marginRight': '8px', 'color': COLORS['info']}),
                        "Comfort Threshold"
                    ], style={'fontSize': '14px', 'fontWeight': '500', 'color': COLORS['dark'], 'marginBottom': '8px', 'display': 'block'}),
                    dcc.Slider(
                        id="comfort-threshold", 
                        min=0.5, max=1.0, step=0.05, value=0.8,
                        marks={0.5: {'label': 'Flexible', 'style': {'color': COLORS['muted'], 'fontSize': '12px'}}, 
                               0.65: {'label': 'Moderate', 'style': {'color': COLORS['muted'], 'fontSize': '12px'}}, 
                               0.8: {'label': 'Standard', 'style': {'color': COLORS['muted'], 'fontSize': '12px'}}, 
                               1.0: {'label': 'Strict', 'style': {'color': COLORS['muted'], 'fontSize': '12px'}}},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    html.Div(id="comfort-value", style={'textAlign': 'center', 'marginTop': '8px', 'color': COLORS['info'], 'fontWeight': '500'}),
                    html.P("Minimum acceptable comfort level for load shifting", 
                           style={'color': COLORS['muted'], 'fontSize': '12px', 'textAlign': 'center', 'marginTop': '8px', 'fontStyle': 'italic'})
                ], style={'marginBottom': '24px'})
            ])
        ], style=CARD_STYLE),
        
        # Action Section
        html.Div([
            html.Button([
                html.I(className="fas fa-cog fa-spin", id="sim-icon", style={'marginRight': '8px'}),
                "Run Simulation"
            ], id="run-button", style=SUCCESS_BUTTON_STYLE),
            
            html.Div(id="simulation-progress", style={'marginTop': '16px'})
        ], style={'textAlign': 'center', 'margin': '24px 0'})
    ])

def create_results_tab():
    return html.Div([
        # Page Header
        html.Div([
            html.H2([
                html.I(className="fas fa-chart-bar", style={'marginRight': '12px', 'color': COLORS['info']}),
                "Simulation Results"
            ], style={'color': COLORS['dark'], 'fontWeight': '600', 'margin': '0 0 8px 0'}),
            html.P("Analyze the performance and financial impact of different tariff structures", 
                   style={'color': COLORS['muted'], 'margin': '0', 'fontSize': '16px'})
        ], style=CARD_STYLE),
        
        # Financial Summary Section
        html.Div([
            html.H3([
                html.I(className="fas fa-euro-sign", style={'marginRight': '8px', 'color': COLORS['success']}),
                "Financial Summary"
            ], style={'color': COLORS['dark'], 'fontWeight': '600', 'marginBottom': '20px'}),
            html.Div(id="financial-summary"),
        ], style=CARD_STYLE),
        
        # Charts Grid
        html.Div([
            # Load Profiles Chart
            html.Div([
                html.H3([
                    html.I(className="fas fa-bolt", style={'marginRight': '8px', 'color': COLORS['primary']}),
                    "Load Profiles"
                ], style={'color': COLORS['dark'], 'fontWeight': '600', 'marginBottom': '20px'}),
                dcc.Graph(id="load-profile-graph", config={'displayModeBar': False}),
            ], style=CARD_STYLE),
            
            # Grid Exchange Chart
            html.Div([
                html.H3([
                    html.I(className="fas fa-exchange-alt", style={'marginRight': '8px', 'color': COLORS['danger']}),
                    "Grid Exchange"
                ], style={'color': COLORS['dark'], 'fontWeight': '600', 'marginBottom': '20px'}),
                dcc.Graph(id="grid-graph", config={'displayModeBar': False}),
            ], style=CARD_STYLE),
            
            # Tariff Comparison Chart
            html.Div([
                html.H3([
                    html.I(className="fas fa-balance-scale", style={'marginRight': '8px', 'color': COLORS['warning']}),
                    "Tariff Comparison"
                ], style={'color': COLORS['dark'], 'fontWeight': '600', 'marginBottom': '20px'}),
                dcc.Graph(id="tariff-comparison-graph", config={'displayModeBar': False}),
            ], style=CARD_STYLE),
        ])
    ])

# Callbacks for dynamic slider value display
@app.callback(
    [Output("buildings-value", "children"),
     Output("residents-value", "children"),
     Output("pv-value", "children"),
     Output("battery-value", "children")],
    [Input("num-buildings", "value"),
     Input("residents-per-building", "value"),
     Input("pv-capacity", "value"),
     Input("battery-capacity", "value")],
    prevent_initial_call=True
)
def update_setup_slider_values(buildings, residents, pv, battery):
    return (
        f"{buildings} buildings",
        f"{residents} residents per building",
        f"{pv} kW solar capacity",
        f"{battery} kWh battery storage"
    )

@app.callback(
    [Output("duration-value", "children"),
     Output("sensitivity-value", "children"),
     Output("comfort-value", "children")],
    [Input("sim-duration", "value"),
     Input("price-sensitivity", "value"),
     Input("comfort-threshold", "value")],
    prevent_initial_call=True
)
def update_simulation_slider_values(duration, sensitivity, comfort):
    return (
        f"{duration} days",
        f"{sensitivity:.1f}",
        f"{comfort:.2f}"
    )

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
                html.Div([
                    html.I(className="fas fa-check-circle", style={'fontSize': '20px', 'marginRight': '8px', 'color': COLORS['success']}),
                    html.Span("Community Successfully Initialized!", style={'fontSize': '16px', 'fontWeight': '600'})
                ], style={'marginBottom': '12px', 'color': COLORS['success']}),
                
                html.Div([
                    html.Div([
                        html.I(className="fas fa-building", style={'marginRight': '6px', 'color': COLORS['primary']}),
                        f"{num_buildings} buildings configured"
                    ], style={'marginBottom': '4px', 'color': COLORS['dark']}),
                    
                    html.Div([
                        html.I(className="fas fa-users", style={'marginRight': '6px', 'color': COLORS['primary']}),
                        f"Total residents: {num_buildings * residents_per_building}"
                    ], style={'marginBottom': '4px', 'color': COLORS['dark']}),
                    
                    html.Div([
                        html.I(className="fas fa-list-check", style={'marginRight': '6px', 'color': COLORS['warning']}),
                        f"Selected tariffs: {', '.join([t.upper() for t in (tariff_types or ['tou'])])}"
                    ], style={'color': COLORS['dark']})
                ], style={'backgroundColor': COLORS['light'], 'padding': '12px', 'borderRadius': '8px', 'border': f'1px solid {COLORS["border"]}'})
            ], style={'textAlign': 'left'})
        except Exception as e:
            return html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-triangle", style={'fontSize': '18px', 'marginRight': '8px', 'color': COLORS['danger']}),
                    "Initialization Failed"
                ], style={'marginBottom': '8px', 'color': COLORS['danger'], 'fontWeight': '600'}),
                
                html.Div([
                    f"Error: {str(e)}"
                ], style={'color': COLORS['muted'], 'fontSize': '14px', 'backgroundColor': COLORS['light'], 'padding': '12px', 'borderRadius': '8px'})
            ])
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
                html.Div([
                    html.I(className="fas fa-check-circle", style={'fontSize': '20px', 'marginRight': '8px', 'color': COLORS['success']}),
                    html.Span("Simulation Completed Successfully!", style={'fontSize': '16px', 'fontWeight': '600'})
                ], style={'marginBottom': '12px', 'color': COLORS['success']}),
                
                html.Div([
                    html.Div([
                        html.I(className="fas fa-calendar", style={'marginRight': '6px', 'color': COLORS['primary']}),
                        f"Duration: {duration} days"
                    ], style={'marginBottom': '4px', 'color': COLORS['dark']}),
                    
                    html.Div([
                        html.I(className="fas fa-trophy", style={'marginRight': '6px', 'color': COLORS['warning']}),
                        f"Best tariff: {results['simulation_summary']['best_tariff'].upper()}"
                    ], style={'marginBottom': '4px', 'color': COLORS['dark']}),
                    
                    html.Div([
                        html.I(className="fas fa-piggy-bank", style={'marginRight': '6px', 'color': COLORS['success']}),
                        f"Total savings: €{results['simulation_summary']['total_savings']:.2f}"
                    ], style={'color': COLORS['dark']})
                ], style={'backgroundColor': COLORS['light'], 'padding': '12px', 'borderRadius': '8px', 'border': f'1px solid {COLORS["border"]}'})
            ])
        except Exception as e:
            return {}, html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-triangle", style={'fontSize': '18px', 'marginRight': '8px', 'color': COLORS['danger']}),
                    "Simulation Failed"
                ], style={'marginBottom': '8px', 'color': COLORS['danger'], 'fontWeight': '600'}),
                
                html.Div([
                    f"Error: {str(e)}"
                ], style={'color': COLORS['muted'], 'fontSize': '14px', 'backgroundColor': COLORS['light'], 'padding': '12px', 'borderRadius': '8px'})
            ])
    return {}, html.Div([
        html.Div([
            html.I(className="fas fa-info-circle", style={'fontSize': '18px', 'marginRight': '8px', 'color': COLORS['warning']}),
            "Please Initialize Community First"
        ], style={'color': COLORS['warning'], 'fontWeight': '600', 'marginBottom': '8px'}),
        
        html.P("Go to the Setup tab and initialize your community before running the simulation.", 
               style={'color': COLORS['muted'], 'fontSize': '14px', 'margin': '0'})
    ])

@app.callback(
    Output("financial-summary", "children"),
    [Input("simulation-results", "data")]
)
def update_financial_summary(results):
    if not results:
        return html.Div([
            html.Div([
                html.I(className="fas fa-info-circle", style={'fontSize': '48px', 'color': COLORS['muted'], 'marginBottom': '16px'}),
                html.H4("No Results Available", style={'color': COLORS['muted'], 'marginBottom': '8px'}),
                html.P("Run a simulation to see financial analysis here.", style={'color': COLORS['muted'], 'fontSize': '14px'})
            ], style={'textAlign': 'center', 'padding': '40px'})
        ])
    
    summary = results['simulation_summary']
    fairness = results['fairness_analysis']
    
    return html.Div([
        # Main Financial Metrics
        html.Div([
            html.Div([
                html.Div([
                    html.I(className="fas fa-trophy", style={'fontSize': '24px', 'color': COLORS['warning'], 'marginBottom': '8px'}),
                    html.H4("Best Tariff", style={'margin': '0', 'color': COLORS['dark'], 'fontWeight': '600'}),
                    html.H3(f"{summary['best_tariff'].upper()}", style={'margin': '4px 0', 'color': COLORS['warning'], 'fontWeight': '700'})
                ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': COLORS['light'], 'borderRadius': '12px', 'margin': '8px'})
            ], style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.Div([
                    html.I(className="fas fa-calculator", style={'fontSize': '24px', 'color': COLORS['info'], 'marginBottom': '8px'}),
                    html.H4("Baseline Cost", style={'margin': '0', 'color': COLORS['dark'], 'fontWeight': '600'}),
                    html.H3(f"€{summary['baseline_cost']:.2f}", style={'margin': '4px 0', 'color': COLORS['info'], 'fontWeight': '700'})
                ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': COLORS['light'], 'borderRadius': '12px', 'margin': '8px'})
            ], style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.Div([
                    html.I(className="fas fa-euro-sign", style={'fontSize': '24px', 'color': COLORS['primary'], 'marginBottom': '8px'}),
                    html.H4("Optimized Cost", style={'margin': '0', 'color': COLORS['dark'], 'fontWeight': '600'}),
                    html.H3(f"€{summary['optimized_cost']:.2f}", style={'margin': '4px 0', 'color': COLORS['primary'], 'fontWeight': '700'})
                ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': COLORS['light'], 'borderRadius': '12px', 'margin': '8px'})
            ], style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.Div([
                    html.I(className="fas fa-piggy-bank", style={'fontSize': '24px', 'color': COLORS['success'], 'marginBottom': '8px'}),
                    html.H4("Total Savings", style={'margin': '0', 'color': COLORS['dark'], 'fontWeight': '600'}),
                    html.H3(f"€{summary['total_savings']:.2f}", style={'margin': '4px 0', 'color': COLORS['success'], 'fontWeight': '700'}),
                    html.P(f"({summary['savings_percent']:.1f}%)", style={'margin': '0', 'color': COLORS['success'], 'fontSize': '14px', 'fontWeight': '500'})
                ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': COLORS['light'], 'borderRadius': '12px', 'margin': '8px'})
            ], style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ], style={'marginBottom': '24px'}),
        
        # Fairness Metrics
        html.Div([
            html.H4([
                html.I(className="fas fa-balance-scale", style={'marginRight': '8px', 'color': COLORS['info']}),
                "Fairness Analysis"
            ], style={'color': COLORS['dark'], 'fontWeight': '600', 'marginBottom': '16px'}),
            
            html.Div([
                html.Div([
                    html.Label("Gini Coefficient", style={'fontSize': '14px', 'fontWeight': '500', 'color': COLORS['muted'], 'marginBottom': '4px', 'display': 'block'}),
                    html.Span(f"{fairness['gini_coefficient']:.3f}", style={'fontSize': '20px', 'fontWeight': '600', 'color': COLORS['dark']})
                ], style={'textAlign': 'center', 'padding': '16px', 'backgroundColor': COLORS['light'], 'borderRadius': '8px', 'margin': '4px', 'width': '32%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Coefficient of Variation", style={'fontSize': '14px', 'fontWeight': '500', 'color': COLORS['muted'], 'marginBottom': '4px', 'display': 'block'}),
                    html.Span(f"{fairness['coefficient_of_variation']:.3f}", style={'fontSize': '20px', 'fontWeight': '600', 'color': COLORS['dark']})
                ], style={'textAlign': 'center', 'padding': '16px', 'backgroundColor': COLORS['light'], 'borderRadius': '8px', 'margin': '4px', 'width': '32%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Cost Range", style={'fontSize': '14px', 'fontWeight': '500', 'color': COLORS['muted'], 'marginBottom': '4px', 'display': 'block'}),
                    html.Span(f"€{fairness['cost_range']:.2f}", style={'fontSize': '20px', 'fontWeight': '600', 'color': COLORS['dark']})
                ], style={'textAlign': 'center', 'padding': '16px', 'backgroundColor': COLORS['light'], 'borderRadius': '8px', 'margin': '4px', 'width': '32%', 'display': 'inline-block'})
            ])
        ], style={'backgroundColor': COLORS['light'], 'padding': '20px', 'borderRadius': '12px', 'border': f'1px solid {COLORS["border"]}'})
    ])

@app.callback(
    Output("load-profile-graph", "figure"),
    [Input("simulation-results", "data")]
)
def update_load_profile(results):
    if not results:
        return go.Figure().update_layout(
            title="Load Profile - No Data Available",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", color=COLORS['muted']),
            annotations=[
                dict(
                    text="Run a simulation to see load profile data",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False,
                    font=dict(size=16, color=COLORS['muted'])
                )
            ]
        )
    
    timestamps = pd.to_datetime(results['timestamps'])
    
    fig = go.Figure()
    
    # Add Total Load trace
    fig.add_trace(go.Scatter(
        x=timestamps, y=results['total_load'],
        name='Total Load',
        line=dict(color=COLORS['primary'], width=3),
        hovertemplate='<b>Total Load</b><br>Time: %{x}<br>Power: %{y:.1f} kW<extra></extra>'
    ))
    
    # Add PV Generation trace
    fig.add_trace(go.Scatter(
        x=timestamps, y=results['pv_generation'],
        name='PV Generation',
        line=dict(color=COLORS['secondary'], width=3),
        fill='tonexty',
        fillcolor=f'rgba({int(COLORS["secondary"][1:3], 16)}, {int(COLORS["secondary"][3:5], 16)}, {int(COLORS["secondary"][5:7], 16)}, 0.2)',
        hovertemplate='<b>PV Generation</b><br>Time: %{x}<br>Power: %{y:.1f} kW<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="Community Load Profile vs PV Generation",
            font=dict(size=18, color=COLORS['dark'], family="Inter, sans-serif")
        ),
        xaxis_title="Time",
        yaxis_title="Power (kW)",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color=COLORS['dark']),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(gridcolor=COLORS['border'], gridwidth=1),
        yaxis=dict(gridcolor=COLORS['border'], gridwidth=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

@app.callback(
    Output("grid-graph", "figure"),
    [Input("simulation-results", "data")]
)
def update_grid_graph(results):
    if not results:
        return go.Figure().update_layout(
            title="Grid Exchange - No Data Available",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", color=COLORS['muted']),
            annotations=[
                dict(
                    text="Run a simulation to see grid exchange data",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False,
                    font=dict(size=16, color=COLORS['muted'])
                )
            ]
        )
    
    timestamps = pd.to_datetime(results['timestamps'])
    grid_exchange = np.array(results['grid_exchange'])
    
    # Separate import and export for different colors
    import_data = np.where(grid_exchange > 0, grid_exchange, 0)
    export_data = np.where(grid_exchange < 0, grid_exchange, 0)
    
    fig = go.Figure()
    
    # Add Import trace (positive values)
    fig.add_trace(go.Scatter(
        x=timestamps, y=import_data,
        fill='tozeroy', name='Grid Import',
        line=dict(color=COLORS['danger'], width=2),
        fillcolor=f'rgba({int(COLORS["danger"][1:3], 16)}, {int(COLORS["danger"][3:5], 16)}, {int(COLORS["danger"][5:7], 16)}, 0.3)',
        hovertemplate='<b>Grid Import</b><br>Time: %{x}<br>Power: %{y:.1f} kW<extra></extra>'
    ))
    
    # Add Export trace (negative values)
    fig.add_trace(go.Scatter(
        x=timestamps, y=export_data,
        fill='tozeroy', name='Grid Export',
        line=dict(color=COLORS['success'], width=2),
        fillcolor=f'rgba({int(COLORS["success"][1:3], 16)}, {int(COLORS["success"][3:5], 16)}, {int(COLORS["success"][5:7], 16)}, 0.3)',
        hovertemplate='<b>Grid Export</b><br>Time: %{x}<br>Power: %{y:.1f} kW<extra></extra>'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS['muted'], line_width=2, opacity=0.7)
    
    fig.update_layout(
        title=dict(
            text="Grid Import/Export Analysis",
            font=dict(size=18, color=COLORS['dark'], family="Inter, sans-serif")
        ),
        xaxis_title="Time",
        yaxis_title="Power (kW)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color=COLORS['dark']),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(gridcolor=COLORS['border'], gridwidth=1),
        yaxis=dict(gridcolor=COLORS['border'], gridwidth=1),
        margin=dict(l=40, r=40, t=60, b=40),
        annotations=[
            dict(
                x=0.02, y=0.98, xref="paper", yref="paper",
                text="<b>📊 Grid Exchange Analysis</b><br>🔴 Positive: Import from Grid<br>🟢 Negative: Export to Grid",
                showarrow=False,
                bgcolor=COLORS['card'],
                bordercolor=COLORS['border'],
                borderwidth=1,
                font=dict(size=12, color=COLORS['dark']),
                align="left"
            )
        ]
    )
    
    return fig

@app.callback(
    Output("tariff-comparison-graph", "figure"),
    [Input("simulation-results", "data")]
)
def update_tariff_comparison(results):
    if not results:
        return go.Figure().update_layout(
            title="Tariff Comparison - No Data Available",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", color=COLORS['muted']),
            annotations=[
                dict(
                    text="Run a simulation to see tariff comparison data",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False,
                    font=dict(size=16, color=COLORS['muted'])
                )
            ]
        )
    
    tariff_comparison = results['tariff_comparison']
    
    tariffs = list(tariff_comparison.keys())
    costs = [tariff_comparison[t]['total_cost'] for t in tariffs]
    savings_pct = [tariff_comparison[t]['cost_savings_percent'] for t in tariffs]
    
    # Create color palette for tariffs
    tariff_colors = [COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['info']]
    
    fig = go.Figure()
    
    # Add Cost bars
    fig.add_trace(go.Bar(
        x=[t.upper() for t in tariffs], 
        y=costs,
        name='Total Cost (€)',
        marker=dict(
            color=tariff_colors[:len(tariffs)],
            line=dict(color=COLORS['dark'], width=1)
        ),
        yaxis='y',
        hovertemplate='<b>%{x}</b><br>Total Cost: €%{y:.2f}<extra></extra>',
        text=[f'€{cost:.2f}' for cost in costs],
        textposition='auto',
        textfont=dict(color='white', size=12, family="Inter, sans-serif")
    ))
    
    # Add Savings percentage line
    fig.add_trace(go.Scatter(
        x=[t.upper() for t in tariffs], 
        y=savings_pct,
        name='Savings (%)',
        mode='markers+lines',
        marker=dict(size=12, color=COLORS['warning'], line=dict(color=COLORS['dark'], width=2)),
        line=dict(color=COLORS['warning'], width=3),
        yaxis='y2',
        hovertemplate='<b>%{x}</b><br>Savings: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="Tariff Performance Comparison",
            font=dict(size=18, color=COLORS['dark'], family="Inter, sans-serif")
        ),
        xaxis_title="Tariff Type",
        yaxis=dict(
            title="Total Cost (€)", 
            side='left',
            titlefont=dict(color=COLORS['primary']),
            tickfont=dict(color=COLORS['primary']),
            gridcolor=COLORS['border'],
            gridwidth=1
        ),
        yaxis2=dict(
            title="Cost Savings (%)", 
            side='right', 
            overlaying='y',
            titlefont=dict(color=COLORS['warning']),
            tickfont=dict(color=COLORS['warning']),
            showgrid=False
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color=COLORS['dark']),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=60, t=60, b=40),
        hovermode='x unified'
    )
    
    return fig

if __name__ == "__main__":
    print("Starting Prosumer Community Simulation Web App...")
    print("Open your browser and go to: http://127.0.0.1:8050")
    app.run_server(debug=True, port=8050)
