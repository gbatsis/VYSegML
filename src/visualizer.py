import os
import random
import json
from unicodedata import name
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from dataHandler import DatasetConstructor,FeatureExtractor

'''
'''
class PlotGenerator():
    def __init__(self):
        self.datasetDir = './VYDataset'

    '''
    '''
    def plotSamples(self,app=False):
        samples = list()
        for ar in os.listdir(self.datasetDir):
            arPath = os.path.join(self.datasetDir,ar)
            arPath = os.path.join(arPath,'Multispectral/images')
            arImgs = os.listdir(arPath)
            sample = random.sample(arImgs,1)[0]
            samples.append(os.path.join(arPath,sample))

        fig = make_subplots(rows=3, cols=4,shared_yaxes=True,shared_xaxes=True,vertical_spacing=0.075,subplot_titles=('RGB Image','Red Edge Band','NIR Band','Ground Truth'))

        fe = FeatureExtractor()

        rows = 5

        for i, sample in enumerate(samples):
            maskPath = '{}/{}/{}/masks/mask_{}'.format(self.datasetDir,sample.split('/')[2],sample.split('/')[3],sample.split('/')[-1].split('_')[-1])

            img = np.load(sample)
            mask = np.load(maskPath)

            rgb = fe.convertToUINT(img[:,:,:3])
            
            re = fe.convertToUINT(img[:,:,3])
            nir = fe.convertToUINT(img[:,:,4])

            imList = [rgb,re,nir,mask]
            for j,im in enumerate(imList):
                col = (j % rows) + 1
                if j == 0:
                    fig.add_trace(go.Image(z=imList[j]),row=i+1, col=col)
                else:
                    fig.add_trace(go.Heatmap(z=imList[j], colorscale='gray',showscale=False), row=i+1, col=col)

        fig.update_yaxes(autorange='reversed', scaleanchor='x', constrain='domain',visible=False,showline=True, linewidth=5, linecolor='white', mirror=True)
        fig.update_xaxes(constrain='domain',visible=False,showline=True, linewidth=5, linecolor='white', mirror=True)
        fig.update_layout(showlegend=False,height=1000, width=1000,
                title_text="Dataset Samples",font=dict(
                family="Courier New, monospace",
                size=25,
                color="gainsboro"),title_x=0.5,
            paper_bgcolor='rgba(0,0,0,0.5)',
            plot_bgcolor='rgba(0,0,0,0.5)',autosize=False)
        
        if app:
            return fig
        else:
            fig.show()
        
    '''
    '''
    def plotComparison(self,app=False):
        dsConstructor = DatasetConstructor()

        results = [            
            (dsConstructor.files['comparisonDF'],"Model Comparison: All Features"),
            (dsConstructor.files['RandomForestComparisonDF'],"Model Comparison: Features selected by Random Forest"),
            (dsConstructor.files['AdaBoostComparisonDF'],'Model Comparison: Features selected by AdaBoost'),
            (dsConstructor.files['anovaComparisonDF'],'Model Comparion: Features selected by ANOVA Method')]

        if app:
            figs=list()
        for res in results:
            resultsDF = pd.read_csv(res[0],index_col=0)
            
            methods = resultsDF['model'].values
            trainingTimes = resultsDF['trainingTime'].values
            testingTimes = resultsDF['testTime'].values
            testAcc =  resultsDF['score'].values
            
            fig = make_subplots(rows=2, cols=1,shared_xaxes=True, subplot_titles=('F1 Score',  'Training & Prediction Time/Samples (μS)'))
            trace1 = go.Bar(x = methods, y = trainingTimes, name = 'Training time/Sample')
            trace2 = go.Bar(x = methods, y = testingTimes, name = 'Prediction time/Sample')
            trace3 = go.Bar(x = methods, y = testAcc, name = 'F1 Score')

            fig.append_trace(trace3, 1,1)
            fig.append_trace(trace1, 2, 1)
            fig.append_trace(trace2,2,1)

            fig.update_layout(title_text=res[1],font=dict(family="Courier New, monospace",size=18,color="gainsboro"),
                barmode='group',bargap=0.30,bargroupgap=0.0,yaxis2 = dict(range=[0,10000]),title_x=0.5,
                paper_bgcolor='rgba(0,0,0,0.5)',
                plot_bgcolor='rgba(0,0,0,0.5)',autosize=False,height=700,width=1200)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='white', mirror=True)
            fig.update_yaxes(showline=True, linewidth=2, linecolor='white', mirror=True)
            
            for i in fig['layout']['annotations']:
                i['font'] = font=dict(
                family="Courier New, monospace",
                size=16,
                color="gainsboro")

            for i in fig['data']:
                i['width']=0.2
            
            if app:
                figs.append(fig)
            else:
                fig.show()

        if app:
            return figs

    '''
    '''
    def plotCV(self,app=False):
        dsConstructor = DatasetConstructor()
        cvResults = json.load(open(dsConstructor.files['cvResults']))
        
        if app:
            return [self.prCurveCV(cvResults,app=True),self.rocCurveCV(cvResults,app=True)]
        else:
            self.prCurveCV(cvResults)
            self.rocCurveCV(cvResults)
    
    '''
    '''
    def rocCurveCV(self,cvResults,app=False):

        H0tprs = list()
        H1tprs = list()
        H0aucs = list()
        H1aucs = list()
        H0meanFpr = H1meanFpr = np.linspace(0, 1, 100)

        for idx in cvResults:
            i='{}'.format(idx)
            H0tpr = np.array(cvResults[i]['rocCurve']['0']['tpr'])
            H0fpr = np.array(cvResults[i]['rocCurve']['0']['fpr'])
            H1tpr = np.array(cvResults[i]['rocCurve']['1']['tpr'])
            H1fpr = np.array(cvResults[i]['rocCurve']['1']['fpr'])

            H0auc = cvResults[i]['rocCurve']['0']['auc']
            H1auc = cvResults[i]['rocCurve']['1']['auc']
            H0aucs.append(H0auc)
            H1aucs.append(H1auc)

            H0tprs.append(np.interp(H0meanFpr,H0fpr,H0tpr))
            H0tprs[-1][0] = 0.0
            H1tprs.append(np.interp(H1meanFpr,H1fpr,H1tpr))
            H1tprs[-1][0] = 0.0

        H0meanTpr = np.mean(H0tprs,axis=0)
        H0meanTpr[-1]=1
        H1meanTpr = np.mean(H1tprs,axis=0)
        H1meanTpr[-1]=1
        H0stdTpr = 2*np.std(H0tprs, axis=0)
        H0upperTpr = np.clip(H0meanTpr+H0stdTpr, 0, 1)
        H0lowerTpr = H0meanTpr-H0stdTpr
        H1stdTpr = 2*np.std(H1tprs, axis=0)
        H1upperTpr = np.clip(H1meanTpr+H1stdTpr, 0, 1)
        H1lowerTpr = H1meanTpr-H1stdTpr

        dataPlot = [
            go.Scatter(x = H0meanFpr,y = H0upperTpr,line = dict(color='darkblue', width=1),hoverinfo = "skip",showlegend = False,name = 'upper'),
            go.Scatter(x = H0meanFpr,y = H0lowerTpr,fill = 'tonexty',fillcolor = 'rgba(52, 152, 219, 0.2)',line = dict(color='darkblue', width=1),hoverinfo = "skip",showlegend = False,name = 'lower'),
            go.Scatter(x = H0meanFpr,y = H0meanTpr,
                line = dict(color='cornflowerblue', width=3,dash='dash'),hoverinfo = "skip",showlegend = True,name='Background - AUC {:.2f}'.format(np.mean(H0aucs))),
            go.Scatter(x = H1meanFpr,y = H1upperTpr,line = dict(color='darkred', width=1),hoverinfo = "skip",showlegend = False,name = 'upper'),
            go.Scatter(x = H1meanFpr,y = H1lowerTpr,fill = 'tonexty',fillcolor = 'rgba(219, 152, 52, 0.2)',line = dict(color='darkred', width=1),hoverinfo = "skip",showlegend = False,name = 'lower'),
            go.Scatter(x = H1meanFpr,y = H1meanTpr,
                line = dict(color='indianred', width=3,dash='dash'),hoverinfo = "skip",showlegend = True,name='Vineyard - AUC {:.2f}'.format(np.mean(H1aucs))),
            go.Scatter(x=[0, 1], y=[0, 1], line=dict(color='gray', width=1, dash='dash'), name='baseline')]

        fig = go.Figure(dataPlot,layout=go.Layout(xaxis=dict(title="False Positive Rate",),  yaxis=dict(title="True Positive Rate",)))
        
        fig.update_layout(title_text='Cross Validation - ROC Curve',font=dict(
                family="Courier New, monospace",
                size=18,
                color="gainsboro"),title_x=0.5,
            paper_bgcolor='rgba(0,0,0,0.5)',
            plot_bgcolor='rgba(0,0,0,0.5)',
            autosize=False,height=700,width=1200)
        
        fig.update_xaxes(showline=True, linewidth=2, linecolor='white', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='white', mirror=True)

        if app:
            return fig
        else:
            fig.show()

    '''
    '''
    def prCurveCV(self,cvResults,app=False):
        meanRecall = np.linspace(0, 1, len(cvResults))
        H0Press = list()
        H1Press = list()
        for idx in cvResults:
            i='{}'.format(idx)
            H0pres = np.array(cvResults[i]['prCurve']['0']['precision'])
            H0recall = np.array(cvResults[i]['prCurve']['0']['recall'])
            H1pres = np.array(cvResults[i]['prCurve']['1']['precision'])
            H1recall = np.array(cvResults[i]['prCurve']['1']['recall'])
            
            H0Press.append(np.interp(meanRecall, H0pres, H0recall))
            H1Press.append(np.interp(meanRecall, H1pres, H1recall))

        H0meanPrecision = np.mean(H0Press, axis=0)
        H1meanPrecision = np.mean(H1Press, axis=0)

        H0stdPrecision = 2*np.std(H0Press, axis=0)
        Η0upperPrecision = np.clip(H0meanPrecision+H0stdPrecision, 0, 1)
        Η0lowerPrecision = H0meanPrecision-H0stdPrecision

        H1stdPrecision = 2*np.std(H1Press, axis=0)
        Η1upperPrecision = np.clip(H1meanPrecision+H1stdPrecision, 0, 1)
        Η1lowerPrecision = H1meanPrecision-H1stdPrecision

        dataPlot = [
            go.Scatter(x = meanRecall,y = Η0upperPrecision,line = dict(color='darkblue', width=1),hoverinfo = "skip",showlegend = False,name = 'upper'),
            go.Scatter(x = meanRecall,y = Η0lowerPrecision,fill = 'tonexty',fillcolor = 'rgba(52, 152, 219, 0.2)',line = dict(color='darkblue', width=1),hoverinfo = "skip",showlegend = False,name = 'lower'),
            go.Scatter(x = meanRecall,y = H0meanPrecision,
                line = dict(color='cornflowerblue', width=3,dash='dash'),hoverinfo = "skip",showlegend = True,name='Background'),
            go.Scatter(x = meanRecall,y = Η1upperPrecision,line = dict(color='darkred', width=1),hoverinfo = "skip",showlegend = False,name = 'upper'),
            go.Scatter(x = meanRecall,y = Η1lowerPrecision,fill = 'tonexty',fillcolor = 'rgba(219, 152, 52, 0.2)',line = dict(color='darkred', width=1),hoverinfo = "skip",showlegend = False,name = 'lower'),
            go.Scatter(x = meanRecall,y = H1meanPrecision,
                line = dict(color='indianred', width=3, dash='dash'),hoverinfo = "skip",showlegend = True,name='Vineyard')]

        fig = go.Figure(dataPlot,layout=go.Layout(xaxis=dict(title="Recall"), yaxis=dict(title="Precission")))
                
        fig.update_layout(title_text='Cross Validation - Precision/Recall Curve',font=dict(
                family="Courier New, monospace",
                size=18,
                color="gainsboro"),title_x=0.5,
            paper_bgcolor='rgba(0,0,0,0.5)',
            plot_bgcolor='rgba(0,0,0,0.5)',
            autosize=False,height=700,width=1200)
        fig.update_xaxes(showline=True, linewidth=2, linecolor='white', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='white', mirror=True)

        if app:
            return fig
        else:
            fig.show()

    '''
    '''    
    def plotReport(self,app=False):
        dsConstructor = DatasetConstructor()
        data = json.load(open(dsConstructor.files['report']))
        cm = data['confusionMatrix']
        classNames = data['classNames']

        if app:
            return [self.confusionMatrix(cm,classNames,app=True),self.prCurve(data['prCurve'],app=True),self.rocCurve(data['rocCurve'],app=True)]
        else:
            self.confusionMatrix(cm,classNames)
            self.prCurve(data['prCurve'])
            self.rocCurve(data['rocCurve'])
    
    '''
    '''
    def rocCurve(self,data,app=False):
        
        colors = ['cornflowerblue', 'indianred']

        plotData = list()

        for i in data:
            className = data[i]['name']
            fpr = np.array(data[i]['fpr'])
            tpr = np.array(data[i]['tpr'])
            thres = np.array(data[i]['threshold'])
            aucScore = data[i]['auc']

            plotData.append(go.Scatter(x=fpr, y=tpr, line=dict(color=colors[int(i)], width=2, dash='dash'),name=f'{className} - ROC - AUC={aucScore:.2f}'))

        plotData.append(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color='gray', width=1, dash='dash'), name='baseline'))
        fig = go.Figure(data=plotData,layout=go.Layout(xaxis=dict(title="False Positive Rate"), yaxis=dict(title="True Positive Rate")))

        fig.update_layout(title_text='Evaluation of Test Dataset: ROC Curve',font=dict(
                family="Courier New, monospace",
                size=18,
                color="gainsboro"),title_x=0.5,
            paper_bgcolor='rgba(0,0,0,0.5)',
            plot_bgcolor='rgba(0,0,0,0.5)',
            autosize=False,height=700,width=1200)
        fig.update_xaxes(showline=True, linewidth=2, linecolor='white', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='white', mirror=True)

        if app:
            return fig
        else:
            fig.show()
        
    '''
    '''
    def prCurve(self,data,app=False):
        
        colors = ['cornflowerblue', 'indianred']

        plotData = list()

        for i in data:
            className = data[i]['name']
            pre = np.array(data[i]['precision'])
            rec = np.array(data[i]['recall'])

            plotData.append(go.Scatter(x=rec, y=pre, line=dict(color=colors[int(i)], width=2, dash='dash'), 
                                        name=className))

        fig = go.Figure(data=plotData,layout=go.Layout(xaxis=dict(title="Recall",),  yaxis=dict(title="Precission")))
        fig.update_layout(title_text='Evaluation of Test Dataset: Precision/Recall Curve',font=dict(
                family="Courier New, monospace",
                size=18,
                color="gainsboro"),title_x=0.5,
            paper_bgcolor='rgba(0,0,0,0.5)',
            plot_bgcolor='rgba(0,0,0,0.5)',
            autosize=False,height=700,width=1200)
        fig.update_xaxes(showline=True, linewidth=2, linecolor='white', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='white', mirror=True)


        if app:
            return fig
        else:
            fig.show()
           
    '''
    '''
    def confusionMatrix(self,cm,classNames,app=False):
        data = go.Heatmap(z=cm, y=classNames, x=classNames)
        annotations = []
        for i, row in enumerate(cm):
            for j, value in enumerate(row):
                annotations.append(
                    {
                        "x": classNames[i],
                        "y": classNames[j],
                        "font": {"color": "white"},
                        "text": str(value),
                        "xref": "x1",
                        "yref": "y1",
                        "showarrow": False,
                    }
                )
        layout = {
            "xaxis": {"title": "Predicted"},
            "yaxis": {"title": "Actual"},
            "annotations": annotations,
        }

        fig = go.Figure(data=data, layout=layout)
        for i in range(len(fig.layout.annotations)):
            fig.layout.annotations[i].font.size = 40

        fig['data'][0]['showscale'] = False
        fig.update_layout( yaxis = dict( tickfont = dict(size=20)),xaxis = dict( tickfont = dict(size=20)),title_text='Evaluation of Test Dataset: Confusion Matrix',font=dict(
                family="Courier New, monospace",
                size=18,
                color="gainsboro"),title_x=0.5,
            paper_bgcolor='rgba(0,0,0,0.5)',
            plot_bgcolor='rgba(0,0,0,0.5)',
            autosize=False,height=700,width=1200)
        fig.update_xaxes(title_font = {"size": 25},showline=True, linewidth=2, linecolor='white', mirror=True)
        fig.update_yaxes(title_font = {"size": 25},title_standoff = 25,showline=True, linewidth=2, linecolor='white', mirror=True)
        
        if app:
            return fig
        else:
            fig.show()

    '''
    '''
    def plotTestPredictions(self,dataToPlot,app=False):
        fe = FeatureExtractor()
        rows = 5

        fig = make_subplots(rows=3, cols=5,shared_yaxes=True,shared_xaxes=True,vertical_spacing=0.005,subplot_titles=('RGB Image','Red Edge Band','NIR Band','Ground Truth','Precition'))


        for i,data in enumerate(dataToPlot):
            img = data[0]
            rgb = fe.convertToUINT(img[:,:,:3])
            re = fe.convertToUINT(img[:,:,3])
            nir = fe.convertToUINT(img[:,:,4])

            mask = data[1]
            predMask = data[2]

            imList = [rgb,re,nir,mask,predMask]
            for j,im in enumerate(imList):
                col = (j % rows) + 1
                if j == 0:
                    fig.add_trace(go.Image(z=imList[j]),row=i+1, col=col)
                else:
                    fig.add_trace(go.Heatmap(z=imList[j],name='m', colorscale='gray',showscale=False), row=i+1, col=col)
                    if j == len(imList)-1:
                        fig.add_annotation(xref="x domain",yref="y domain",x=0.5, y=1.2, showarrow=False,
                            text='Dice Score = {:.2}'.format(data[-1]),font=dict(
                            family="Courier New, monospace",
                            size=15,
                            color="gainsboro"), row=i+1, col=col)

        fig.update_yaxes(autorange='reversed', scaleanchor='x', constrain='domain',visible=False,showline=True, linewidth=5, linecolor='white', mirror=True)
        fig.update_xaxes(constrain='domain',visible=False,showline=True, linewidth=5, linecolor='white', mirror=True)
        fig.update_layout(showlegend=False,height=1000, width=1000,
                title_text="Predictions from Test Dataset",font=dict(
                family="Courier New, monospace",
                size=25,
                color="gainsboro"),title_x=0.5,
            paper_bgcolor='rgba(0,0,0,0.5)',
            plot_bgcolor='rgba(0,0,0,0.5)',autosize=False)
        
        if app:
            return fig
        else:
            fig.show()

    '''
    '''
    def plotDeployment(self,rgb,predImg,dice,app=False):
        predFig = make_subplots(rows=3, cols=4,shared_yaxes=True,shared_xaxes=True,vertical_spacing=0.075)

        predFig.add_trace(go.Image(z=rgb),row=1, col=1)
        predFig.add_annotation(xref="x domain",yref="y domain",x=0.5, y=1.2, showarrow=False,
                            text='RGB Image',font=dict(
                            family="Courier New, monospace",
                            size=25,
                            color="white"), row=1, col=1)

        predFig.add_trace(go.Image(z=predImg),row=1, col=2)
        predFig.add_annotation(xref="x domain",yref="y domain",x=0.5, y=1.2, showarrow=False,
                            text='Prediction: Dice = {:.2}'.format(dice),font=dict(
                            family="Courier New, monospace",
                            size=25,
                            color="white"), row=1, col=2)

        predFig.update_layout(autosize=False,template='plotly_dark',height=2000, width=2000)
        predFig.update_xaxes(visible=False,showline=True, linewidth=5, linecolor='white', mirror=True)
        predFig.update_yaxes(visible=False,showline=True, linewidth=5, linecolor='white', mirror=True)
        
        predFig.update_layout(title_x=0.5,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)')

        if app:
            return predFig
        else:
            predFig.show()

        
