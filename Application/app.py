import sys
import dash
import dash_bootstrap_components as dbc

from dash import html,dcc
from dash.dependencies import Input, Output

sys.path.append('./src')
from dataHandler import DatasetConstructor
from mlMethods import MLFramework
from visualizer import PlotGenerator

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)
dsConstructor = DatasetConstructor()
plotter = PlotGenerator()
mlFw = MLFramework()

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "#303030",

}

CONTENT_STYLE = {
    "padding": "2rem 1rem",
    "background-color": "black",
    'background-image':'url(assets/background.jpg)',
    'background-repeat': 'no-repeat',
    'verticalAlign':'middle',
    'textAlign': 'center',
    'position':'absolute',
    'width':'100%',
    'height':'350%',
    'top':'0px',
    'padding-left':'25%'
}

sidebar = html.Div(
    [
        html.H2("Main Menu", style={'textAlign':'center','color':'antiquewhite'},className="display-4"),
        dbc.Nav(
            [
                dbc.NavLink("Home",style={'textAlign':'center','color':'antiquewhite'}, href="/", active="exact"),
                dbc.NavLink("Dataset Sample",style={'textAlign':'center','color':'antiquewhite'}, href="/page-1", active="exact"),
                dbc.NavLink("Comparison & Feature Selection",style={'textAlign':'center','color':'antiquewhite'}, href="/page-2", active="exact"),
                dbc.NavLink("Cross Validation",style={'textAlign':'center','color':'antiquewhite'}, href="/page-3", active="exact"),
                dbc.NavLink("Test Classification Report",style={'textAlign':'center','color':'antiquewhite'}, href="/page-4", active="exact"),
                dbc.NavLink("Live Segmentation",style={'textAlign':'center','color':'antiquewhite'}, href="/page-5", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

app.layout = html.Div([
    dcc.Location(id="url"),
    content,
    sidebar
])

@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/":
        return [
                html.H1('UAV Imagery & Machine Learning | Vineyard Segmentation',
                        style={'textAlign':'right','color':'antiquewhite'}),

                dbc.Container(
                    [
                        dbc.Row(dbc.Col(html.Div(
                children=[
                    dcc.Markdown('''
                    The emergence of new technological methods in the field of Robotic Systems and Computer Vision has led to the more frequent implementation of Remote Sensing (RS) applications for the automated control of agricultural activities towards Precision Agriculture. In particular, the advent of Unarmed Aerial Vehicles (UAV) offered to the domain experts the convenience to develop RS applications because these systems are a low cost and flexible solution. In this work, a study of Semantic Segmentation for vineyard recognition is presented by combining Multispectral UAV imagery data, Machine Learning (ML) algorithms, Feature Extraction and Feature Selection Methods. The dataset which was used is here https://github.com/Cybonic/DL_vineyard_segmentation_study.git. Concerning Feature Extraction methods, Vegetation Indices and Texture from images were extracted. In order to fit (the Non-Deep) ML methods, it is essential that Feature Extraction Methods should be combined with an efficient sampling method to convert the entire set of images to a pixel-based Dataset. Different ML methods were compared in terms of training and prediction time and their performance and Gaussian Naive Bayes (GNB) was the most efficient method. Despite the fact that GNB was not accurate in the prediction of data containing the entire set of extracted features, accuracy of this method increased after Feature Selection. More precisely, F1 score of GNB combined with Features selected by a tree-based ensemble classifier (Random Forest, AdaBoost) was competitive in comparison with Random Forest, AdaBoost and SVM-RBF and considering its agility during prediction, GNB was finally selected.
                    '''),
                    ]
                        ))),
                                ],style={'color':'antiquewhite','text-align':'justify','padding-top':'20%','padding-bottom':'20%','padding-left':'10%', 'padding-right':'10%','background-color':'rgba(0,0,0,0.5)'})

                ]
    elif pathname == "/page-1":
        fig=plotter.plotSamples(app=True)
        return [
                dcc.Graph(id='dsample',
                         figure=fig,style={'width': '20%','padding-left':'5%', 'padding-right':'25%'})
                ]
    elif pathname == "/page-2":
        figs = plotter.plotComparison(app=True)
        figList = [dcc.Graph(id='compFig{}'.format(i+1),figure=figs[i],style={'padding-left':'5%'}) for i in range(len(figs))]
        return figList
    elif pathname == "/page-3":
        figs = plotter.plotCV(app=True)
        figList = [dcc.Graph(id='compFig{}'.format(i+1),figure=figs[i],style={'padding-left':'5%','padding-right':'5%'}) for i in range(len(figs))]
        return figList
    elif pathname == "/page-4":
        figs = plotter.plotReport(app=True)
        figList = [dcc.Graph(id='compFig{}'.format(i+1),figure=figs[i],style={'padding-left':'5%','padding-right':'5%'}) for i in range(len(figs))]
        figList.insert(0,dcc.Graph(id='predFig',figure=mlFw.testPredictions(app=True),style={'padding-left':'5%','padding-right':'5%'}))
        return figList
    elif pathname == "/page-5":
        return [
                dbc.Container(
                    [
                        dcc.Store(id="store"),
                        html.H2("Click the button below to generate a random RGB Image \
                        from the Test Dataset and to predict segmentation mask. \
                        Ground Truth is shown with original image.",style={'fontSize': 20,'textAlign':'center','color':'ghostwhite','padding-left':'20%','padding-right':'20%'}),
                        dbc.Row(dbc.Col(dbc.Button("Pick a Random Image",style={'fontSize': 20,'textAlign':'center','color':'ghostwhite'}, id='button', n_clicks=0, color="secondary", size='lg'),
                        style={'textAlign': 'center'})),                        
                        dbc.Row(html.Div(id="prediction", className="p-4"),style={'textAlign': 'center'})
                    ]
                )
                
            ]

    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

@app.callback(
    Output("prediction", "children"),
    Input("store", "prediction"),
)
def render_tab_content(prediction):
    return dcc.Graph(id='depImg',figure=prediction,style={'padding-left':'5%'})

@app.callback(Output("store", "prediction"), [Input("button", "n_clicks")])
def generate_graphs(n):
    predFig = mlFw.deployModel(app=True)
    return predFig

if __name__=='__main__':
    app.run_server(port= 5001,debug=False)