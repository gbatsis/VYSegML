import os
import time
import random
import json
import numpy as np
import pandas as pd
import pickle
import tifffile
import cv2 

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel,SelectKBest,f_classif
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline

from dataHandler import DatasetConstructor, FeatureExtractor
from visualizer import PlotGenerator

class MLFramework():
    def modelSelection(self,trainDF,testDF,selection=True,selected=None):
        print('|---|---| Model Selection based on Training Time, Prediction Time & F1 Score')

        dsConstructor = DatasetConstructor()

        X_train = trainDF.drop(['label','imgName','mode'], axis=1).values
        y_train = trainDF['label'].values

        X_test = testDF.drop(['label','imgName','mode'], axis=1).values
        y_test = testDF['label'].values

        scaler = StandardScaler()
        scaler.fit(X_train)      
        X_train = scaler.transform(X_train)          
        X_test = scaler.transform(X_test)

        methods = ['SVM','GaussianNB','KNN','RandomForest','AdaBoostClassifier']
        testAcc, trainingTimes, testingTimes = [], [], []
        for method in methods:
            print('|---|---|    Training {}...'.format(method))
            if method=='SVM':
                clf = SVC()
            elif method=='GaussianNB':
                clf = GaussianNB()
            elif method=='KNN':
                clf = KNeighborsClassifier()
            elif method=='RandomForest':
                clf = RandomForestClassifier()
                if selection:
                   rfSelection = self.modelFeatureSelection(method,clf,X_train,y_train,trainDF)
            elif method=='AdaBoostClassifier':
                clf = AdaBoostClassifier()
                if selection:
                   adaSelection = self.modelFeatureSelection(method,clf,X_train,y_train,trainDF)

            # Training procedure.
            startTime = time.time()
            clf.fit(X_train, y_train)
            fittingTime = time.time() - startTime
            trainingTimes.append(fittingTime*1e9/X_train.shape[0])

            print('|---|---|    Predicting using {}...'.format(method))

            # Prediction procedure.
            startTime = time.time()
            y_pred = clf.predict(X_test)
            predictionTime = time.time() - startTime
            testingTimes.append(predictionTime*1e9/X_test.shape[0])
            testAcc.append(metrics.f1_score(y_test, y_pred, average='macro'))

            print('|---|---|    {}'.format('-'*30))

        results = pd.DataFrame(
            {
                'model':methods,
                'trainingTime':trainingTimes,
                'testTime':testingTimes,
                'score':testAcc
            })

        if selection:
            results.to_csv(dsConstructor.files['comparisonDF'])
    
            rfFts = {
                'features':rfSelection.tolist()
            }
        
            with open(dsConstructor.files['rfFeatures'], 'w') as f1:
                json.dump(rfFts,f1)

            adaFts = {
                    'features':adaSelection.tolist()
            }

            with open(dsConstructor.files['adaFeatures'], 'w') as f2:
                json.dump(adaFts,f2)
        
        else:
            results.to_csv(dsConstructor.files['{}ComparisonDF'.format(selected)])

    '''
    '''
    def modelFeatureSelection(self,method,clf,X_train,y_train,trainDF):
        sfm = SelectFromModel(clf)
        sfm.fit(X_train,y_train)

        usefull = sfm.get_support()
        usefull = np.where(usefull == True)[0]

        informativeFeatures = trainDF.iloc[:,usefull].columns
        print('|---|---|        {} selected these features:'.format(method))
        time.sleep(5)
        print(json.dumps(informativeFeatures.tolist(),indent=8))
        
        return informativeFeatures.values

    '''
    '''
    def statisticfeatureSelection(self,trainDF,k):
        dsConstructor = DatasetConstructor()

        X_train = trainDF.drop(['label','imgName','mode'], axis=1).values
        y_train = trainDF['label'].values

        scaler = StandardScaler()
        scaler.fit(X_train)      
        X_train = scaler.transform(X_train)          

        fs = SelectKBest(score_func=f_classif, k=k)

        fs.fit(X_train, y_train)
        fs.feature_names_in_ = trainDF.drop(['label','imgName','mode'], axis=1).columns

        anovaSelection = fs.get_feature_names_out().values
        
        anovaFts = {
                'features':anovaSelection.tolist()
        }

        with open(dsConstructor.files['anovaFeatures'], 'w') as f2:
            json.dump(anovaFts,f2)

    '''
    '''
    def crossVal(self,datasetDF,bestFeatures):
        print('|---|---| Perform Cross Validation...')
        dsConstructor = DatasetConstructor()
        rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)
        cvResults = {}
        idx = 0
        f1Scores = list()
        thrs = list()
        for i_train, i_test in rkf.split(datasetDF.index.values):
            trainDF = datasetDF.iloc[i_train]
            valDF = datasetDF.iloc[i_test]

            X_train = trainDF.drop(['label','imgName'], axis=1).values
            y_train = trainDF['label'].values

            X_val = valDF.drop(['label','imgName'], axis=1).values
            y_val = valDF['label'].values

            scaler = StandardScaler()
            scaler.fit(X_train)      
            X_train = scaler.transform(X_train)        
            X_val = scaler.transform(X_val)

            clf = GaussianNB()
            clf.fit(X_train,y_train)

            y_pred_prob = clf.predict_proba(X_val)
            y_pred = clf.predict(X_val)
            
            f1Scores.append(metrics.f1_score(y_val, y_pred, average='macro'))

            classNames = ['background','vineyard']

            prCurveData = dict()

            for ic, c in enumerate(classNames):
                pre, rec, thres = metrics.precision_recall_curve(y_val, y_pred_prob[:, classNames.index(c)], pos_label=ic)
                prCurveData[ic] = {
                    'name':c,
                    'precision':pre.tolist(),
                    'recall':rec.tolist(),
                    'threshold':thres.tolist()
                }

            rocCurveData = dict()

            for ic, c in enumerate(classNames):
                fpr, tpr, thres = metrics.roc_curve(y_val, y_pred_prob[:, classNames.index(c)], pos_label=ic)
                
                if ic==1:
                    tfpRate = tpr - fpr
                    ix = np.argmax(tfpRate)
                    thrs.append(thres[ix])

                aucScore = metrics.auc(fpr, tpr)
                rocCurveData[ic] = {
                    'name':c,
                    'fpr':fpr.tolist(),
                    'tpr':tpr.tolist(),
                    'threshold':thres.tolist(),
                    'auc':aucScore
                }

            cvResults[idx] = {
                'classNames' : classNames,
                'prCurve' : prCurveData,
                'rocCurve' : rocCurveData
            }

            idx+=1

        meanF1 = np.mean(np.array(f1Scores))
        stdF1 = np.std(np.array(f1Scores))

        print(f'|---|---|    Performance -> Mean F1: {meanF1:.3f} | Sigma F1: {stdF1:.3f} | 95% Conf: {meanF1-2*stdF1:.3f} - {meanF1+2*stdF1:.3f}')
        print('|---|---|    {}'.format('-'*30))
        
        th = np.mean(np.array(thrs))

        tfDict = {
            'th':th,
            'features':bestFeatures
        }
        
        print('|---|---|    Mean value of optimal threshold is: {:.3}'.format(th))

        with open(dsConstructor.files['cvResults'], 'w') as f:
            json.dump(cvResults,f)

        with open(dsConstructor.files['tf'], 'w') as ft:
            json.dump(tfDict,ft)

        print('|---|---|    Training using all data and save the classifier...')

        X = datasetDF.drop(['label','imgName'], axis=1).values
        y = datasetDF['label'].values

        pipe = Pipeline([
        ('scale', StandardScaler()),
        ('clf', GaussianNB())])

        pipe.fit(X,y)
        
        with open(dsConstructor.files['gaussianNB'], 'wb') as f:
            pickle.dump(pipe, f)

        return th



    def classificationReport(self,unseenDF,th):
        dsConstructor = DatasetConstructor()
        print('|---|---| Final Evaluation & Classification Report:')
        
        clf = pickle.load(open(dsConstructor.files['gaussianNB'], 'rb'))

        y_test = unseenDF['label'].values
        X_test = unseenDF.drop(['label','imgName','mode'], axis=1).values    

        y_pred_prob = clf.predict_proba(X_test)

        y_pred = (y_pred_prob[:,1] >= th).astype('int')

        print(metrics.classification_report(y_test, y_pred))

        y_pred_prob = clf.predict_proba(X_test)
        
        classNames = ['background','vineyard']

        prCurveData = dict()

        for ic, c in enumerate(classNames):
            pre, rec, thres = metrics.precision_recall_curve(y_test, y_pred_prob[:, classNames.index(c)], pos_label=ic)
            prCurveData[ic] = {
                'name':c,
                'precision':pre.tolist(),
                'recall':rec.tolist(),
                'threshold':thres.tolist()
            }

        rocCurveData = dict()

        for ic, c in enumerate(classNames):
            fpr, tpr, thres = metrics.roc_curve(y_test, y_pred_prob[:, classNames.index(c)], pos_label=ic)
            aucScore = metrics.auc(fpr, tpr)
            rocCurveData[ic] = {
                'name':c,
                'fpr':fpr.tolist(),
                'tpr':tpr.tolist(),
                'threshold':thres.tolist(),
                'auc':aucScore
            }

        classificationReport = {
            'classNames' : classNames,
            'confusionMatrix' : metrics.confusion_matrix(y_test, y_pred).tolist(),
            'prCurve' : prCurveData,
            'rocCurve' : rocCurveData
        }
     
        with open(dsConstructor.files['report'], 'w') as f:
            json.dump(classificationReport,f)
    
    '''
    '''
    def dice_coef(self,y_true, y_pred, epsilon=1e-6):
        y_true_flatten = np.asarray(y_true).astype(np.bool)
        y_pred_flatten = np.asarray(y_pred).astype(np.bool)

        if not np.sum(y_true_flatten) + np.sum(y_pred_flatten):
            return 1.0

        return (2. * np.sum(y_true_flatten * y_pred_flatten)) /\
            (np.sum(y_true_flatten) + np.sum(y_pred_flatten) + epsilon)


    '''
    '''
    def testPredictions(self,app=False):
        dsConstructor = DatasetConstructor()
        fe = FeatureExtractor()
        plotter = PlotGenerator()

        clf = pickle.load(open(dsConstructor.files['gaussianNB'], 'rb'))
        tf = json.load(open(dsConstructor.files['tf']))
        
        files = [f for f in Path(dsConstructor.datasetDir).rglob('*{}*.npy'.format(dsConstructor.deployPrefix))]

        imgList = random.sample(files,3)
        
        dataToPlot = list()

        for imgPath in imgList:
            maskPath = [f for f in Path(dsConstructor.datasetDir).rglob('*mask_{}'.format(imgPath.name.split('_')[-1]))][0]
            
            img = np.load(imgPath)
            mask = np.load(maskPath)
            data = fe.extractor((img,mask,'test'),sampling=False)
            data = data[data.columns.intersection(tf['features'])]

            X_test = data.values    
        
            y_pred_prob = clf.predict_proba(X_test)

            y_pred = (y_pred_prob[:,1] >= tf['th']).astype('int')

            predMask = y_pred.reshape(mask.shape)

            dataToPlot.append((img,mask,predMask,self.dice_coef(mask,predMask)))

        plotter.plotTestPredictions(dataToPlot,app)

        if app:
            return plotter.plotTestPredictions(dataToPlot,app)


    '''
    '''
    def deployModel(self,app=False):
        dsConstructor = DatasetConstructor()
        fe = FeatureExtractor()
        plotter = PlotGenerator()

        clf = pickle.load(open(dsConstructor.files['gaussianNB'], 'rb'))
        tf = json.load(open(dsConstructor.files['tf']))
        
        imList = [imgPath for imgPath in Path(dsConstructor.appImgs).rglob('*.tiff')]
        imgPath = random.sample(imList,1)[0]

        name = imgPath.name[3:-5]
        maskPath = os.path.join(dsConstructor.appImgs,'gt{}.npy'.format(name))
        msPath = os.path.join(dsConstructor.appImgs,'ms{}.npy'.format(name))

        rgb = tifffile.imread(imgPath)
        mask = np.load(maskPath)

        mask = np.where(mask==255, 1, 0)
        msImage = np.load(msPath)
        
        data = fe.extractor((msImage,mask,'test'),sampling=False)
        data = data[data.columns.intersection(tf['features'])]

        X_test = data.values    
        
        y_pred_prob = clf.predict_proba(X_test)

        y_pred = (y_pred_prob[:,1] >= tf['th']).astype('int')

        predMask = y_pred.reshape(mask.shape)
        color = np.array([255,255,0], dtype='uint8')

        predMasked = np.where(predMask[...,None], color, rgb)
        predImg = cv2.addWeighted(rgb, 0.8, predMasked, 0.2,0)
        dice = self.dice_coef(mask,predMask)

        if app:
            fig = plotter.plotDeployment(rgb,predImg,dice,app=True)
            return fig
        else:
            plotter.plotDeployment(predImg,dice)
        

        