from flask import Flask, render_template, request, jsonify, send_file, app, g
import pandas as pd

from pathlib import Path
import sys
import matplotlib
import os

ui_path = Path(__file__).parent.resolve()
path_to_add = str(ui_path.parent.parent.parent)
sys.path.append(path_to_add)




from tabularmagic.wizard.api import Wizard


wizard: Wizard = None


def chat(msg: str) -> str:
    """
    Chat function that processes natural language queries on the uploaded dataset.
    """
    global wizard
    if wizard is None:
        return "No dataset uploaded. Please upload a dataset first."
    
    else:
        return wizard.chat(
            msg,
            which="single"
        )
    
def get_analysis():
    return wizard._canvas_queue.get_analysis()

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_dataset():
    """
    Handle dataset upload and store it for the chat function.
    """
    global wizard
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        # Read the uploaded CSV file
        uploaded_data = pd.read_csv(file, index_col=0)
        wizard = Wizard(uploaded_data, test_size=0.2)

        return jsonify({"message": "Dataset uploaded successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat_route():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    response_message = chat(user_message)
    return jsonify({"response": response_message})


@app.route("/analysis", methods=["GET"])
def get_analysis_history():
    """
    Retrieve the current analysis history (figures and tables).
    """
    if wizard is None:
        return jsonify({"error": "No dataset uploaded. Please upload a dataset first."}), 400
    
    try:
        analysis_items = get_analysis()
        items = []
        for path in analysis_items:
            path_obj = Path(path)
            if path_obj.exists():
                if path_obj.suffix == ".png":
                    items.append({"file_name": path_obj.name, "file_type": "figure", "file_path": str(path_obj)})
                elif path_obj.suffix == ".pkl":
                    # Load the DataFrame and convert to HTML
                    df = pd.read_pickle(path_obj)
                    html_table = df.to_html(classes="table", index=True)
                    items.append({"file_name": path_obj.name, "file_type": "table", "content": html_table})
            else:
                app.logger.warning(f"File not found: {path}")
        return jsonify(items)
    except Exception as e:
        app.logger.error(f"Error retrieving analysis history: {str(e)}")
        return jsonify({"error": "Failed to retrieve analysis history"}), 500


    
@app.route('/analysis/file/<filename>', methods=['GET'])
def serve_file(filename):
    """
    Serve static files (figures) from the analysis queue.
    """
    if wizard is None:
        return jsonify({"error": "No dataset uploaded. Please upload a dataset first."}), 400
    
    analysis_items = get_analysis()
    for path in analysis_items:
        if Path(path).name == filename:
            file_path = Path(path)
            if file_path.exists():
                return send_file(file_path)
    
    return jsonify({"error": f"File '{filename}' not found."}), 404

def get_wizard():
    if 'wizard' not in g:
        g.wizard = None
    return g.wizard

@app.teardown_appcontext
def cleanup_wizard(exception=None):
    g.pop('wizard', None)

# Run the app
if __name__ == "__main__":
    matplotlib.use("Agg")
    app.run(debug=True)


