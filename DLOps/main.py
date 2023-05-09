import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import transformers
import torch

# Define the label representation
label_rep = {
    'ISTJ': 0, 'ISFJ': 1, 'INFJ': 2, 'INTJ': 3, 'ISTP': 4, 'ISFP': 5, 'INFP': 6, 'INTP': 7,
    'ESTP': 8, 'ESFP': 9, 'ENFP': 10, 'ENTP': 11, 'ESTJ': 12, 'ESFJ': 13, 'ENFJ': 14, 'ENTJ': 15
}

# Load the pre-trained model and tokenizer
model = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_rep),
                                                      output_attentions=False,
                                                      output_hidden_states=False)
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model_path = "trained_model.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Initialize the app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1('MBTI Personality Type Predictor'),
    html.H2("Gaurav Sangwan, Sachin Gaur, Ritish Khichi"),
    dcc.Textarea(
        id='input-text',
        placeholder='Enter your text here...',
        value='',
        style={'width': '100%', 'height': 200}
    ),
    html.Button('Predict', id='submit-btn', n_clicks=0),
    html.Div(id='output')
])

# Define the callback
@app.callback(
    Output('output', 'children'),
    Input('submit-btn', 'n_clicks'),
    State('input-text', 'value')
)
def predict_personality(n_clicks, input_text):
    if n_clicks > 0:
        
        # Encode the input text using the tokenizer
        encoded_input = tokenizer(input_text, return_tensors='pt')

        # Pass the encoded input through the model
        with torch.no_grad():
            outputs = model(**encoded_input)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
        predicted_personality = list(label_rep.keys())[list(label_rep.values()).index(predicted_label)]

        return f'Predicted personality type: {predicted_personality}'

if __name__ == '__main__':
    app.run_server(debug=True)
