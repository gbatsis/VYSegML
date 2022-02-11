import os
import json
import numpy as np

from dataHandler import DatasetConstructor
from mlMethods import MLFramework
from visualizer import PlotGenerator

'''
'''
def featureChoiceMenu():
    print('|---|---| Select Feature Selection Method according to comparison plot:')
    options = {
        1 : 'Random Forest Features',
        2 : 'AdaBoost Features',
        3 : 'ANOVA Test Features'
    }

    for key in options.keys():
        print ('    |---|---| {}: {}'.format(key,options[key]))

'''
'''
def runEverything(plotter,dsConstructor,mlFw):

    devDataDF, unseenDF = dsConstructor.dsGenerator(display=True)

    msk = np.random.rand(len(devDataDF)) < 0.8
    trainDF = devDataDF[msk]
    valDF = devDataDF[~msk]
    
    plotter.plotSamples()
    
    # Compare Multiple Classifiers using all features.
    mlFw.modelSelection(trainDF,valDF)
    
    # Compare Multiple Classifiers using feature selection.
    rfSelection = json.load(open(dsConstructor.files['rfFeatures']))['features']
    adaSelection = json.load(open(dsConstructor.files['adaFeatures']))['features']

    mlFw.statisticfeatureSelection(devDataDF,int(len(rfSelection)+len(adaSelection)/2))
    anovaSelection = json.load(open(dsConstructor.files['anovaFeatures']))['features']
    
    mlFw.modelSelection(trainDF[trainDF.columns.intersection(rfSelection+['mode','imgName','label'])]
                        ,valDF[valDF.columns.intersection(rfSelection+['mode','imgName','label'])],
                        selection=False,
                        selected='RandomForest')

    mlFw.modelSelection(trainDF[trainDF.columns.intersection(adaSelection+['mode','imgName','label'])]
                        ,valDF[valDF.columns.intersection(adaSelection+['mode','imgName','label'])],
                        selection=False,
                        selected='AdaBoost')
    
    mlFw.modelSelection(trainDF[trainDF.columns.intersection(anovaSelection+['mode','imgName','label'])]
                        ,valDF[valDF.columns.intersection(anovaSelection+['mode','imgName','label'])],
                        selection=False,
                        selected='anova')
    
    plotter.plotComparison()
    
    while(True):
            
        featureChoiceMenu()
        option = ''
        try:
            option = int(input('    |---|---| Enter your choice: '))
        except:
            print('    |---|---| Wrong input. Please enter a number ...')

        if option == 1:
            bestFeatures = rfSelection
            break
        elif option == 2:
            bestFeatures = adaSelection
            break
        elif option == 3:
            bestFeatures = anovaSelection
            break
        else:
            print('    |---|---| Invalid option. Please enter a number between 1 and 4.')

    # Perform Cross Validation.
    th = mlFw.crossVal(devDataDF[devDataDF.columns.intersection(bestFeatures+['imgName','label'])],bestFeatures)
    
    plotter.plotCV()
    
    # Classification of unseen data.
    unseenDF = unseenDF[unseenDF.columns.intersection(bestFeatures+['mode','imgName','label'])]
    mlFw.classificationReport(unseenDF,th)    
    plotter.plotReport()
    mlFw.testPredictions()

'''
'''
def plotEverything(plotter,mlFw):
    plotter.plotSamples()
    plotter.plotComparison()
    plotter.plotCV()
    plotter.plotReport()
    mlFw.testPredictions()


'''
'''
def printMenu():

    print ('    |---|---| Running Options:')

    options = {
        1 : 'Run Everything',
        2 : 'Display Results',
        0 : 'Exit'
    }

    for key in options.keys():
        print ('    |---|---| {}: {}'.format(key,options[key]))


'''
'''
def main():
    print('|---|---| Machine Learning -- MS UAV Imagery -- Vineyard Segmentation.')
    plotter = PlotGenerator()
    dsConstructor = DatasetConstructor()
    mlFw = MLFramework()
    
    fileList = list()
    for f in dsConstructor.files:
        fileList.append(os.path.isfile(dsConstructor.files[f]))

    while(True):
        if all(fileList):
            print('|---|---| All files are available.')
            printMenu()
            option = ''
            try:
                option = int(input('    |---|---| Enter your choice: '))
            except:
                print('    |---|---| Wrong input. Please enter a number ...')
    
            if option == 1:
                runEverything(plotter,dsConstructor,mlFw)
            elif option == 2:
                plotEverything(plotter,mlFw)
            elif option == 0:
                exit()
            else:
                print('    |---|---| Invalid option. Please enter a number between 1 and 4.')
        else:
            runEverything(plotter,dsConstructor,mlFw)
            break
'''
'''
if __name__ == "__main__":
    main()