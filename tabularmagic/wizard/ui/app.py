import dash
from dash import html, dcc, callback, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from pathlib import Path
import sys
import base64
import io
import datetime
import pandas as pd

path_to_src = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(path_to_src))

from tabularmagic.wizard._src.agents.eda_agent import build_eda_agent
from tabularmagic.wizard._src.agents.linear_regression_agent import (
    build_linear_regression_agent,
)


# Placeholder function for LLM interaction
def f(message):
    return f"LLM response to: {message}"


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3("Chat with LLM"),
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            id="chat-history",
                                            style={
                                                "height": "400px",
                                                "overflow-y": "auto",
                                            },
                                        ),
                                        dbc.Input(
                                            id="user-input",
                                            placeholder="Type your message...",
                                            type="text",
                                        ),
                                        dbc.Button(
                                            "Send",
                                            id="send-button",
                                            color="primary",
                                            className="mt-2",
                                        ),
                                    ]
                                )
                            ]
                        ),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        html.H3("File Upload"),
                        dcc.Upload(
                            id="upload-data",
                            children=html.Div(
                                ["Drag and Drop or ", html.A("Select Files")]
                            ),
                            style={
                                "width": "100%",
                                "height": "60px",
                                "lineHeight": "60px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "5px",
                                "textAlign": "center",
                                "margin": "10px",
                            },
                            multiple=False,
                        ),
                        html.Div(id="output-data-upload"),
                    ],
                    width=6,
                ),
            ]
        )
    ],
    fluid=True,
)


@callback(
    Output("chat-history", "children"),
    Output("user-input", "value"),
    Input("send-button", "n_clicks"),
    State("user-input", "value"),
    State("chat-history", "children"),
    prevent_initial_call=True,
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


@callback(
    Output("output-data-upload", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("upload-data", "last_modified"),
)
def update_output(content, name, date):
    if content is not None:
        content_type, content_string = content.split(",")
        decoded = base64.b64decode(content_string)
        try:
            if "csv" in name:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            elif "xls" in name:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))
            else:
                return html.Div(
                    ["Unsupported file type. Please upload a CSV or Excel file."]
                )

            return html.Div(
                [
                    html.H5(f"File: {name}"),
                    html.H6(f"Last modified: {datetime.datetime.fromtimestamp(date)}"),
                    dash_table.DataTable(
                        data=df.head().to_dict("records"),
                        columns=[{"name": i, "id": i} for i in df.columns],
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "height": "auto",
                            "minWidth": "100px",
                            "width": "150px",
                            "maxWidth": "180px",
                            "whiteSpace": "normal",
                        },
                    ),
                ]
            )
        except Exception as e:
            print(e)
            return html.Div(["There was an error processing this file."])


if __name__ == "__main__":
    app.run_server(debug=True)
