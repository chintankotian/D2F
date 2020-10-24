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
        self.batchCounter = 0

        #  Loads the keras datagenerator obj and reads the files from the images folder
        self.dataGeneratorObj = ImageDataGenerator(rescale = self.imgScaling)
        print('Started loading dataset from '+self.dataPath)
        self.dataGenerator = self.dataGeneratorObj.flow_from_directory(self.dataPath, target_size = self.imgSize, batch_size = self.batchSize)
        print('Finished loading dataset')



    def batchReset(self):
        # Reset the batch index of the datagenerator iterator
        self.dataGenerator.reset()


    def nextBatch(self):
        # Return the next batch of images
        reset = False
        images, _ = self.dataGenerator.next()

        if(self.batchCounter > self.dataGenerator.batch_index):
            reset = True
            self.batchCounter = -1

        self.batchCounter += 1

        return self.applyTransform(images), reset


 
    def getBatchIndex(self):
        return self.dataGenerator.batch_index


    def applyRescale1(self,switch = True):
        # Toggles the rescale flag
        self.rescale1 = switch


    def applyTransform(self, data):
        # converts pixel values from [0,1] to [-1,1]
        data = np.array(data)

        if(self.rescale1):
            data = 2*data - 1

        return data


    def reverseTransform(self, data):
        # converts pixel values from [-1,1] to [0,1]
        data = np.array(data)

        if(self.rescale1):
            data = (data + 1)/2

        return data

    


if(__name__ == "__main__"):
    print('normal scaling')
    loader = datasetLoader('../dataset/img_align_celeba', imgSize=(128,128))   
    batchImages,_ = loader.nextBatch()
    print("Min = "+ str(batchImages.min()))
    print("max = "+ str(batchImages.max()))

    plt.imshow(batchImages[0])
    plt.show()

    print("*"*5)

    print("Standard Scaling")
    loader.applyRescale1()
    batchImages,_ = loader.nextBatch()
    print("Min = "+ str(batchImages.min()))
    print("max = "+ str(batchImages.max()))

    print("*"*5)
    print("Reverse transform function")
    newData = loader.reverseTransform(batchImages)
    print("Min = "+str(newData.min()))
    print("Max = "+str(newData.max()))

    plt.imshow(newData[0])
    plt.show()

    # print("Batch imdex at start = "+str(loader.getBatchIndex()))
    # reset  = False
    # while(not reset):
    #     _,reset = loader.nextBatch()
    #     print("Batch index in loop = "+str(loader.getBatchIndex()))
        
    # print("-"*10)
    # print("Batch imdex at end = "+str(loader.getBatchIndex()))
