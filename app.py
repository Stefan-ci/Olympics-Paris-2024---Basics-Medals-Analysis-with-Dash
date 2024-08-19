import dash
import logging
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import dcc, Input, html, Output


# Using logging instead of print()
logging.basicConfig(level=logging.ERROR)

# Declaring required constants to use as global vars
REQUIRED_COLUMNS = ["NOC", "Gold", "Silver", "Bronze", "Total"]
MEDAL_TYPES = ["Gold", "Silver", "Bronze"]
MAX_ROWS_DISPLAY = 10
DATASET_ORIGIN = "https://www.kaggle.com/datasets/muhammadehsan000/olympic-games-medal-dataset-1994-2024/data"
MEDAL_INFO = {
    "Gold": {"title": "Gold medals", "color": "Gold", "column": "Gold"},
    "Silver": {"title": "Silver medals", "color": "Silver", "column": "Silver"},
    "Bronze": {"title": "Bronze medals", "color": "Bronze", "column": "Bronze"}
}


# Loading data
def load_data():
    """
    Load and preprocess Olympic medals data from a CSV file.

    Returns:
        DataFrame: A pandas DataFrame containing the preprocessed data.
                    Returns None if an error occurs or required columns are missing.
    """
    # All exceptions aren't necessary
    try:
        data = pd.read_csv("assets/2024_Paris_Olympics_Nations_Medals.csv")
    except FileNotFoundError:
        logging.error("Error: The file was not found.")
        return None
    except pd.errors.EmptyDataError:
        logging.error("Error: The file is empty.")
        return None
    except pd.errors.ParserError:
        logging.error("Error: The file could not be parsed.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None
    
    # make sure that all required aren't missing
    if not all(column in data.columns for column in REQUIRED_COLUMNS):
        logging.error("Error: Missing one or more required columns.")
        return None
    
    
    data = pd.read_csv("assets/2024_Paris_Olympics_Nations_Medals.csv")
    data["Gold"] = pd.to_numeric(data["Gold"], errors="coerce")
    data["Silver"] = pd.to_numeric(data["Silver"], errors="coerce")
    data["Bronze"] = pd.to_numeric(data["Bronze"], errors="coerce")
    data["Total"] = pd.to_numeric(data["Total"], errors="coerce")
    
    # or use a dynamic way to declare numeric cols
    # numeric_columns = ["Gold", "Silver", "Bronze", "Total"] # more cols can be added in the future if needed
    # for column in numeric_columns:
    #     data[column] = pd.to_numeric(data[column], errors="coerce")
    
    return data



# Get data
data = load_data()


# Web app page header
def page_header_row():
    """
    Returns page HTML header
    
    Returns:
        dbc.Row: A Dash Bootstrap Component Row containing the page header.
    """
    
    if data is None or not all(k in data for k in REQUIRED_COLUMNS):
        raise ValueError("Data is missing required keys")
    
    return dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("Olympics Paris 2024 Medals Analysis"),
                html.H5([
                    html.Span(f"Nations: {len(data["NOC"].to_list())}"),
                    html.Span(",", className="me-2"),
                    html.Span(f"Total medals: {data["Total"].sum()}"),
                ]),
                html.P([
                    html.Span(f"Gold: {data["Gold"].sum()}"),
                    html.Span(",", className="me-2"),
                    
                    html.Span(f"Silver: {data["Silver"].sum()}"),
                    html.Span(",", className="me-2"),
                    
                    html.Span(f"Bronze: {data["Bronze"].sum()}"),
                ])
            ], className="text-center mt-2"),
        ])
    ])



# An Alert() section to call to contribution and describe
def call_to_contribution_row():
    """ An Alert() to describe dataset and call to contribution """
    return dbc.Row([
        dbc.Col([
            dbc.Alert([
                html.P([
                    html.Div([
                        html.Strong("Dataset origin:", className="me-2"),
                        html.Span([
                            html.Span("Muhammad Ehsan", className="me-2"),
                            html.A("From Kaggle", href=DATASET_ORIGIN, className="alert-link", target="_blank"),
                            html.Em("(Last fetch: 19/08/2024)", className="ms-2 text-muted small"),
                        ])
                    ]),
                    html.Div([
                        html.Strong("Dataset columns:", className="me-2"),
                        html.Span([html.Span(f"{str(col)} - ") for col in data.columns])
                    ]),
                    html.Div([
                        html.Strong("NOC:", className="me-2"),
                        html.Span("Stands for Nations Of Competition")
                    ]),
                    html.Div("You could create a new column in the dataset with the full name of each country:", className="mt-3"),
                    html.Ul([
                        html.Li("CIV -> Ivory Coast (Côte d'Ivoire)"),
                        html.Li("USA -> United States Of America"),
                        html.Li("FRA -> France"),
                        html.Li("…"),
                    ])
                ]),
            ], color="primary"),
        ], className="col-md-8 offset-md-2")
    ], className="py-3")



# Page content row (for medals graphs)
def medals_graphs_row():
    """ Graph to display graphs of medals """
    row = dbc.Row([
        # Display medals count by type (can be filtered by "Gold", "Silver", "Bronze")
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Graph of medals - Filter by type", className="card-title"),
                    dcc.Dropdown(
                        id="medals-by-country",
                        value=None,
                        placeholder="Select a medal type",
                        options=[{"label": medal_type, "value": medal_type} for medal_type in MEDAL_TYPES],
                    ),
                    dcc.Graph(id="medals-by-country-graph"),
                ])
            ]),
        ], width=12, className="mt-4"),
        
        
        # Display graph of medals type (can be filtered by country)
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Graph of all medals by type - Filter by country", className="card-title"),
                    dcc.Dropdown(
                        id="medals-type-by-country-filter",
                        value=None,
                        placeholder="Select a country",
                        options=[{"label": country, "value": country} for country in data["NOC"].unique()],
                    ),
                    dcc.Graph(id="medals-by-type-graph"),
                ])
            ]),
        ], width=5, className="mt-4"),
        
        
        # Data table to visualize the first rows
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Overview of the dataset", className="card-title"),
                    dbc.Table([
                            html.Thead(
                                html.Tr([html.Th(col, className="fw-bold") for col in data.columns])
                            ),
                            html.Tbody([
                                html.Tr([
                                    html.Td(data.iloc[i][col]) for col in data.columns
                                ]) for i in range(min(len(data), MAX_ROWS_DISPLAY)) # change it if you want
                            ]),
                        ],
                        bordered=True,
                        # dark=True,
                        hover=True,
                        responsive=True,
                        striped=True,
                    ),
                ])
            ]),
        ], width=7, className="mt-4"),
    ])
    return row






# Initializing Dash() app
app = dash.Dash(
    __name__,
    title="Olympics Paris 2024 Medals Basic Analysis - Dashboard",
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

# Displaying all page content (layout) in a Container() component
app.layout = dbc.Container([
    # Page header row
    page_header_row(),
    
    # Call to contribution row
    call_to_contribution_row(),
    
    # Graphs of medals row
    medals_graphs_row(),
], className="mb-5", fluid=True)







# callback of medal type graph
@app.callback(
    Output("medals-by-country-graph", "figure"), # "figure" from plotly
    Input("medals-by-country", "value")
)

def update_medals_by_country(selected_type):
    if selected_type:
        if not isinstance(selected_type, str):
            return dbc.Alert(f"Invalid input type. Must be a string and match one of {str(MEDAL_TYPES)}", color="error")
        
        elif selected_type not in MEDAL_TYPES:
            return dbc.Alert("Selected medal type isn't available", color="error")
    
    
    # else
    fig_title = "All types of medals"
    fig_color = "Total"
    y_axe_values = "Total"
    
    if selected_type in MEDAL_INFO:
        medal = MEDAL_INFO[selected_type]
        fig_title = medal["title"]
        fig_color = medal["color"]
        y_axe_values = medal["column"]
        filtered_df = data.loc[data[medal["column"]] > 0]
    else:
        fig_title = "All types of medals"
        fig_color = "Total"
        y_axe_values = "Total"
        filtered_df = data # entire df
    
    
    if filtered_df.empty: # empty df
        return dbc.Alert("No data available", color="warning")
    
    num_bins = min(10, len(filtered_df))
    fig = px.histogram(
        data_frame=filtered_df,
        x="NOC",
        y=y_axe_values,
        color=fig_color,
        title=fig_title,
        nbins=num_bins
    )
    return fig










# callback of medals by country graph
@app.callback(
    Output("medals-by-type-graph", "figure"), # "figure" from plotly
    Input("medals-type-by-country-filter", "value")
)
def update_medals_by_type_graph(selected_country):
    ALL_NATIONS = data["NOC"].to_list()
    
    if selected_country:
        if not isinstance(selected_country, str):
            return dbc.Alert(f"Invalid input type. Must be a string and match one of {str(ALL_NATIONS)}", color="error")
        
        elif selected_country not in ALL_NATIONS:
            return dbc.Alert("Selected country type isn't available", color="error")
        
        # else (everything is OK)
        filtered_df = data.loc[data["NOC"] == selected_country] # df based on the selected country
    
    else: # no country selected
        filtered_df = data # entire df
    
    if filtered_df.empty: # empty df
        return dbc.Alert("No data available", color="warning")
    
    labels = MEDAL_TYPES
    values = [filtered_df[label].sum() for label in labels]
    
    fig_title = f"All medal types - Total: {sum(values)}"
    if selected_country:
        fig_title = f"Medal types of {selected_country} - Total: {sum(values)}"
    
    fig = px.pie(values=values, names=labels, title=fig_title, color_discrete_sequence=["#636EFA", "#ef5538"])
    return fig






# running the app (with Flask)
if __name__ == "__main__":
    app.run_server(debug=True)
