import os
import sys
import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append('./GLCM')        
from fast_glcm import *

class DatasetConstructor():
    def __init__(self):
        self.datasetDir = './VYDataset'
        self.runTimeDataFolder = './RunTime'
        self.runTimeDataInit = self.runTimeInitializer()
        self.files = {
            'DatasetDF' : './RunTime/pixelDataset.csv',
            'rfFeatures':'./RunTime/RandomForestSelectedFeatures.json',
            'adaFeatures':'./RunTime/AdaBoostSelectedFeatures.json',
            'anovaFeatures':'./RunTime/AnovaSelectedFeatures.json',
            'comparisonDF':'./RunTime/comparisonDF.csv',
            'RandomForestComparisonDF':'./RunTime/RFcomparisonDF.csv',
            'AdaBoostComparisonDF':'./RunTime/ADAcomparisonDF.csv',
            'anovaComparisonDF':'./RunTime/ANOVAcomparisonDF.csv',
            'cvResults' : './RunTime/cvResults.json',
            'tf' : './RunTime/tf.json',
            'gaussianNB': './RunTime/clf.pkl',
            'report' : './RunTime/report.json'
                               
        }
        self.deployPrefix = 'img_3'
        self.appImgs = './Application/App Images'

    def runTimeInitializer(self):
        os.makedirs(self.runTimeDataFolder,exist_ok=True)
        return True

    '''
        Dataset Generator.
    '''
    def dsGenerator(self,display=False,app=False):
        if app:
            datasetDF = pd.read_csv(self.files['DatasetDF'],index_col=0)
            return datasetDF[datasetDF['mode'] == 'development'].reset_index(drop=True), datasetDF[datasetDF['mode'] == 'deploy'].reset_index(drop=True)

        if display:
            print('|---|---| Building pixel-based dataset:')
            print('|---|---|    Performing Feature Extraction and an efficient pixel sampling')
            print('|---|---|    Please wait...')

        fetchedData = self.dataFetching()

        fe = FeatureExtractor()
        datasetDF = fe.performExtraction(fetchedData)

        datasetDF = datasetDF.assign(mode='development')
        datasetDF.loc[datasetDF['imgName'].str.match(self.deployPrefix), 'mode'] = 'deploy'

        datasetDF = datasetDF.sample(frac=1).reset_index(drop=True)

        if display:
            print('|---|---| Dataset Head:')
            print(datasetDF.head())
            print('|---|---|    Number of Features: {}'.format(datasetDF.shape[1]-3))

            print('|---|---|    Number of Samples: {} for training/validation and {} for deployment.'.format(datasetDF[datasetDF['mode'] == 'development'].shape[0],
                                                                                                            datasetDF[datasetDF['mode'] == 'deploy'].shape[0]))
        datasetDF.to_csv(self.files['DatasetDF'])
        return datasetDF[datasetDF['mode'] == 'development'].reset_index(drop=True), datasetDF[datasetDF['mode'] == 'deploy'].reset_index(drop=True)


    '''
        Fetch data from memory: (Image Array, Mask Array, Name String)
    '''
    def dataFetching(self):
        dataArray = list()
        for imgPath in Path(self.datasetDir).rglob('*.npy'):
            if imgPath.parts[3] == 'images':
                maskPath = './{}/{}/{}/masks/mask_{}'.format(imgPath.parts[0],imgPath.parts[1],imgPath.parts[2],os.path.basename(imgPath).split('_')[1])
                image = np.load(imgPath)
                mask = np.load(maskPath)
                name = imgPath.name.split('.')[0]
                currentData = (image,mask,name)
                dataArray.append(currentData)
        return dataArray


'''
'''
class FeatureExtractor():
    def performExtraction(self,data):        
        pool = mp.Pool()
        pool = mp.Pool(mp.cpu_count())
        dataDFs = pool.map(self.extractor,data)
        pool.close()

        return pd.concat(dataDFs).reset_index(drop=True)

    '''
    '''
    def extractor(self,data,sampling=True):
        samples = 250

        img = data[0]
        label = data[1]

        spectralFeatures = self.generateSpectralFeatures(img)
        features = spectralFeatures.reshape(spectralFeatures.shape[0]*spectralFeatures.shape[1], spectralFeatures.shape[2])
        
        mask = label.reshape(label.shape[0]*label.shape[1], 1)
        mask = np.where(mask==255, 1, 0)

        red = img[:,:,0]
        green = img[:,:,1]
        blue = img[:,:,2]
        re = img[:,:,3]
        nir = img[:,:,4]
        thermal = img[:,:,5]
        
        textureFeatures = np.concatenate((self.getGLCM(self.convertToUINT(red)),
                            self.getGLCM(self.convertToUINT(green)),
                            self.getGLCM(self.convertToUINT(blue)),
                            self.getGLCM(self.convertToUINT(re)),
                            self.getGLCM(self.convertToUINT(nir)),
                            self.getGLCM(self.convertToUINT(thermal))),axis=1)

        features = np.hstack((features,textureFeatures))
        
        if sampling:
            samplesIdx = self.sampling(mask,samples)
            features = features[samplesIdx]
            mask = mask[samplesIdx]

        featureNames = ['R','G','B','RE','NIR','TH',
                        'RENDVI','NDVI','GNDVI','BNDVI','RRVI','RGVI','RBVI','NDGRI','NDGBI',
                        'R_mean','R_std','R_contrast','R_homogeneity','R_ASM','R_energy','R_max','R_entropy',
                        'G_mean','G_std','G_contrast','G_homogeneity','G_ASM','G_energy','G_max','G_entropy',
                        'B_mean','B_std','B_contrast','B_homogeneity','B_ASM','B_energy','B_max','B_entropy',
                        'RE_mean','RE_std','RE_contrast','RE_homogeneity','RE_ASM','RE_energy','RE_max','RE_entropy',
                        'NIR_mean','NIR_std','NIR_contrast','NIR_homogeneity','NIR_ASM','NIR_energy','NIR_max','NIR_entropy',
                        'T_mean','T_std','T_contrast','T_homogeneity','T_ASM','T_energy','T_max','T_entropy']
        
        dataDF = pd.DataFrame(data=features,columns=featureNames)
        dataDF['label'] = mask
        dataDF['imgName'] = data[2]

        return dataDF

    '''
    '''
    def convertToUINT(self,img):
        return (img * 255 / np.max(img)).astype('uint8')

    '''
        #    https://github.com/delgadocc/RHBV/blob/master/pipeline.py
    '''
    def NVI(self,bandA, bandB, minVal, maxVal):
        np.seterr(divide='ignore', invalid='ignore')
        mask = np.greater(bandB + bandA, 0)
        VI = np.choose(mask, (0, (bandA - bandB) / (bandA + bandB)))
        VI[VI[:, :] > maxVal] = maxVal
        VI[VI[:, :] <= minVal] = minVal

        return VI
    
    '''
    '''
    
    def sampling(self,mask,samples):
        indicesH0 = np.where(mask == 0)[0]
        indicesH1 = np.where(mask == 1)[0]
        
        while(1):
            
            if indicesH1.shape[0]>samples/2:
                sampleIndicesH1 = np.random.choice(indicesH1,int(samples/2))
                sampleIndicesH0 = np.random.choice(indicesH0,int(samples/2))
                samplesIdx = np.append(sampleIndicesH0,sampleIndicesH1)
            else:
                sampleIndicesH0 = np.random.choice(indicesH0,int(samples-(indicesH1.shape[0]/2)))
                sampleIndicesH1 = np.random.choice(indicesH1,int(indicesH1.shape[0]/2))
                samplesIdx = np.append(sampleIndicesH0,indicesH1)
   

            difH0 = np.diff(sampleIndicesH0)
            condH0 = (np.any(abs(difH0) < 5)) and (np.any(abs(difH0-240) < 5)) and (np.any(abs(difH0-240*2) < 5)) and (np.any(abs(difH0-240*3) < 5)) and (np.any(abs(difH0-240*4) < 5)) and (np.any(abs(difH0-240*3) < 5))
            
            difH1 = np.diff(sampleIndicesH1)
            condH1 = (np.any(abs(difH1) < 5)) and (np.any(abs(difH1-240) < 5)) and (np.any(abs(difH1-240*2) < 5)) and (np.any(abs(difH1-240*3) < 5)) and (np.any(abs(difH1-240*4) < 5)) and (np.any(abs(difH1-240*3) < 5))
            
            if (condH0 == False) and (condH1==False):
                break

        return samplesIdx
    
    '''
    '''
    def generateSpectralFeatures(self,img):
        spectralFeatures = np.zeros((img.shape[0],img.shape[1],15))

        red = self.convertToUINT(img[:,:,0])
        green = self.convertToUINT(img[:,:,1])
        blue = self.convertToUINT(img[:,:,2])
        re = self.convertToUINT(img[:,:,3])
        nir = self.convertToUINT(img[:,:,4])
        thermal = self.convertToUINT(img[:,:,5])

        RENDVI = self.NVI(nir,re, -1, 1)
        NDVI = self.NVI(nir,red, -1, 1)
        GNDVI = self.NVI(nir,green, -1, 1)
        BNDVI =  self.NVI(nir,blue, -1, 1)
        RRVI = self.NVI(re,red, -1, 1)
        RGVI = self.NVI(re,green, -1, 1)
        RBVI = self.NVI(re,blue, -1, 1)
        NDGRI = self.NVI(green,red, -1, 1)
        NDGBI = self.NVI(green,blue, -1, 1)
        
        spectralFeatures[:,:,0] = red
        spectralFeatures[:,:,1] = green
        spectralFeatures[:,:,2] = blue
        spectralFeatures[:,:,3] = re
        spectralFeatures[:,:,4] = nir
        spectralFeatures[:,:,5] = thermal

        spectralFeatures[:,:,6] = RENDVI
        spectralFeatures[:,:,7] = NDVI
        spectralFeatures[:,:,8] = GNDVI
        spectralFeatures[:,:,9] = BNDVI
        spectralFeatures[:,:,10] = RRVI 
        spectralFeatures[:,:,11] = RGVI
        spectralFeatures[:,:,12] = RBVI
        spectralFeatures[:,:,13] = NDGRI
        spectralFeatures[:,:,14] = NDGBI

        return spectralFeatures

    '''
    '''
    def getGLCM(self,img):
        nbit = 8
        ks = 5
        mi, ma = 0, 255
        texture = np.stack((fast_glcm_mean(img, mi, ma, nbit, ks),
                                fast_glcm_std(img, mi, ma, nbit, ks),
                                fast_glcm_contrast(img, mi, ma, nbit, ks),
                                fast_glcm_homogeneity(img, mi, ma, nbit, ks),
                                fast_glcm_ASM(img, mi, ma, nbit, ks)[0],
                                fast_glcm_ASM(img, mi, ma, nbit, ks)[1],
                                fast_glcm_max(img, mi, ma, nbit, ks),
                                fast_glcm_entropy(img, mi, ma, nbit, ks)),axis=2)

        textureFeatures = texture.reshape(texture.shape[0]*texture.shape[1], texture.shape[2])

        return textureFeatures

