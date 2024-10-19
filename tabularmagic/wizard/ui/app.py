import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from pathlib import Path
import sys




path_to_src = Path(__file__).resolve().parent.parent / "_src"
sys.path.append(str(path_to_src))



from _src.agents.eda_agent import build_eda_agent
from _src.agents.linear_regression_agent import build_linear_regression_agent






# Placeholder function for LLM interaction
def f(message):
    return f"LLM response to: {message}"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])




app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3("Chat with LLM"),
            dbc.Card([
                dbc.CardBody([
                    html.Div(id="chat-history", style={"height": "400px", "overflow-y": "auto"}),
                    dbc.Input(id="user-input", placeholder="Type your message...", type="text"),
                    dbc.Button("Send", id="send-button", color="primary", className="mt-2"),
                ])
            ]),
        ], width=6),
        dbc.Col([
            html.H3("Right Side (To be added later)"),
            html.Div("Content will be added here in the future.")
        ], width=6),
    ])
], fluid=True)




@callback(
    Output("chat-history", "children"),
    Output("user-input", "value"),
    Input("send-button", "n_clicks"),
    State("user-input", "value"),
    State("chat-history", "children"),
    prevent_initial_call=True
)
def update_chat(n_clicks, user_input, chat_history):
    if user_input:
        user_message = html.Div(f"You: {user_input}")
        llm_response = html.Div(f"LLM: {f(user_input)}")
        
        if chat_history is None:
            chat_history = []
        
        chat_history.extend([user_message, llm_response])
        
        return chat_history, ""
    return chat_history, user_input

if __name__ == "__main__":
    app.run_server(debug=True)





