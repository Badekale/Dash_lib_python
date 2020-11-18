# We start with the import of standard ML librairies
import pandas as pd
import numpy as np
import math

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

# We add all Plotly and Dash necessary librairies
import plotly.graph_objects as go

import dash
import dash_core_components as dcc
import dash_html_components as html
# import dash_daq as daq
from dash.dependencies import Input, Output

import pickle

with open("Iris.pkl", 'rb') as file:  
    Pickled_LR_Model = pickle.load(file)

Pickled_LR_Model



# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash()

# The page structure will be:
#    Features Importance Chart
#    <H4> Feature #1 name
#    Slider to update Feature #1 value
#    <H4> Feature #2 name
#    Slider to update Feature #2 value
#    <H4> Feature #3 name
#    Slider to update Feature #3 value
#    <H2> Updated Prediction
#    Callback fuction with Sliders values as inputs and Prediction as Output

# We apply basic HTML formatting to the layout
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

app.layout = html.Div(style={'textAlign': 'center', 'width': '800px', 'font-family': 'Verdana' },
                        children=[
                        # The same logic is applied to the following names / sliders
                        html.H1(children="Iris Dataset"),
                        

                        # The same logic is applied to the following names / sliders
                        html.H4(children="sepal length (cm)"),
                        dcc.Input(
                            id='X1_slider',
                            type="number",
                            min=1,
                            max=10,
                        #  debounce delay the reaction of the value entered
                        #  tab or enter as to be pressed before the number is updated
                            debounce=True,
                            value = 3,
                            step=0.1,
                            placeholder="sepal length (cm)"),

                        # dcc.Slider(
                        #     id='X1_slider',
                        #     min=1,
                        #     max=10,
                        #     step=0.1,
                        #     value=5.8,
                        #     marks={i: '{:.1f}cm'.format(i) for i in np.linspace(1, 10,15)}
                        # ),

                        html.H4(children='sepal width (cm)',style={'textAlign': 'left'}),

                        dcc.Slider(
                            id='X2_slider',
                            min=1,
                            max=6,
                            step=0.1,
                            value=3,
                            marks={i: '{:.1f}cm'.format(i) for i in np.linspace(1,6,10)}
                        ),
                        html.H4(children='petal length (cm)'),

                        dcc.Slider(
                            id='X3_slider',
                            min=1,
                            max=8,
                            step=0.1,
                            value=3,
                            marks={i: '{:.1f}cm'.format(i) for i in np.linspace(1,8,15)}
                        ),
                        html.H4(children='petal width (cm)'),
                        dcc.Slider(

                            id='X4_slider',
                            min=0.1,
                            max=5,
                            step=0.1,
                            value=3,
                            marks={i: '{:.1f}cm'.format(i) for i in np.linspace(1,5,10)}),
                        
                        # The predictin result will be displayed and updated here
                        html.H2(id="prediction_result")

                    ])

# The callback function will provide one "Ouput" in the form of a string (=children)
@app.callback(Output(component_id="prediction_result",component_property="children"),
# The values correspnding to the three sliders are obtained by calling their id and value property
              [Input("X1_slider","value"), Input("X2_slider","value"), Input("X3_slider","value"),Input("X4_slider","value")])

# The input variable are set in the same order as the callback Inputs
def update_prediction(X1_slider, X2_slider, X3_slider,X4_slider):

    # We create a NumPy array in the form of the original features
    # ["Pressure","Viscosity","Particles_size", "Temperature","Inlet_flow", "Rotating_Speed","pH","Color_density"]
    # Except for the X1, X2 and X3, all other non-influencing parameters are set to their mean
    input_X = np.array([X1_slider,X2_slider,X3_slider,X4_slider]).reshape(1,-1)        
    
    # Prediction is calculated based on the input_X array
    prediction = Pickled_LR_Model.predict(input_X)
    probability = Pickled_LR_Model.predict_proba(input_X)
    probability=np.amax(probability)
    probability = 100 * probability
    if prediction==0:
        flower= 'Setosa'
    elif prediction==1:
        flower= 'Versicolor'
    else:
        flower= 'Virginica'

    # And retuned to the Output of the callback function
    return f'The feature indicated have {probability:.0f}% probability of it been a {flower} iris flower'
# print('The feature indicated have', color.BOLD + f'{probability:.0f}%' + color.End ,f'probability of it been a {flower} iris flower')
if __name__ == "__main__":
    app.run_server(debug=True)
    # app.run_server(debug=True,port=8080,host='0.0.0.0')