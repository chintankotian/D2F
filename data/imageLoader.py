from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt 

class datasetLoader():

    def __init__(self, dataPath, rescale1 = False,batchSize = 32, imgScaling = 1./255, imgSize = (128, 128)):
        self.dataPath = dataPath
        self.batchSize = batchSize
        self.imgScaling = imgScaling
        self.rescale1 = rescale1
        self.imgSize = imgSize

        self.dataGeneratorObj = ImageDataGenerator(rescale = self.imgScaling)
        print('Started loading dataset from '+self.dataPath)
        self.dataGenerator = self.dataGeneratorObj.flow_from_directory(self.dataPath, target_size = self.imgSize, batch_size = self.batchSize)
        print('Finished loading dataset')

    def nextBatch(self):
        images, _ = self.dataGenerator.next()

        if(self.rescale1):
            images = 2*images - 1

        return images

    def applyRescale1(self,switch = True):
        self.rescale1 = switch


if(__name__ == "__main__"):
    print('normal scaling')
    loader = datasetLoader('../dataset/img_align_celeba', imgSize=(1080,1080))   
    batchImages = loader.nextBatch()
    print("Min = "+ str(batchImages.min()))
    print("max = "+ str(batchImages.max()))

    plt.imshow(batchImages[0])
    plt.show()

    loader.applyRescale1()
    batchImages = loader.nextBatch()
    print("Min = "+ str(batchImages.min()))
    print("max = "+ str(batchImages.max()))

