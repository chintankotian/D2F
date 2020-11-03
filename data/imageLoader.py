from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img, save_img
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

    
    def loadImage(self, imgPath, imgSize = None,applyTransform = True):
        # loads single images and applies pixel transforms
        if(not imgSize):
            imgSize = self.imgSize

        img = load_img(imgPath)
        img = img_to_array(img)
        if(applyTransform):
            img = self.applyTransform(img * self.imgScaling)
        
        return img

    
    def saveImage(self, img, imgName, imgPath = "./imageLoaderImages"):
        # Converts img array to img and saves in the given directory
        if(not os.path.isdir(imgPath)):
            os.mkdir(imgPath)
        img = np.array(img)
        imgMax = img.max()
        imgMin = img.min()

        if(imgMin < 0):
            img = self.reverseTransform(img)
        
        if(imgMax > 1):
            img = img * (1/self.imgScaling)
        
        savePath = os.path.join(imgPath,imgName)
        save_img(savePath, img)


    


if(__name__ == "__main__"):
    print('normal scaling')
    loader = datasetLoader('../dataset/img_align_celeba', imgSize=(128,128))   
    batchImages,_ = loader.nextBatch()
    print("Min = "+ str(batchImages.min()))
    print("max = "+ str(batchImages.max()))

    # plt.imshow(batchImages[0])
    # plt.show()

    # print("*"*5)

    # print("Standard Scaling")
    # loader.applyRescale1()
    # batchImages,_ = loader.nextBatch()
    # print("Min = "+ str(batchImages.min()))
    # print("max = "+ str(batchImages.max()))

    # print("*"*5)
    # print("Reverse transform function")
    # newData = loader.reverseTransform(batchImages)
    # print("Min = "+str(newData.min()))
    # print("Max = "+str(newData.max()))

    # plt.imshow(newData[0])
    # plt.show()

    # print("Batch imdex at start = "+str(loader.getBatchIndex()))
    # reset  = False
    # while(not reset):
    #     _,reset = loader.nextBatch()
    #     print("Batch index in loop = "+str(loader.getBatchIndex()))
        
    # print("-"*10)
    # print("Batch imdex at end = "+str(loader.getBatchIndex()))

    print("*"*5)
    print("Testing loading and saving images")

    print("1 - ")
    print('Loading images without scaling')
    path = os.path.join("../dataset/img_align_celeba/img_align_celeba",os.listdir("../dataset/img_align_celeba/img_align_celeba")[0] )
    unscaledImg = loader.loadImage(path, applyTransform=False)
    print("Unscaled image max = ",unscaledImg.max())
    print("Unscaled image min = ",unscaledImg.min())

    print("2 - ")
    print('Loading images with scaling')
    # path = os.path.join("./datset/img_align_celeba/img_align_celeba",os.listdir("./datset/img_align_celeba/img_align_celeba")[0] )
    scaledImg = loader.loadImage(path, applyTransform=True)
    print("scaled image max = ",scaledImg.max())
    print("scaled image min = ",scaledImg.min())

    print("3 - ")
    print('Loading images with normal scaling')
    # path = os.path.join("./datset/img_align_celeba/img_align_celeba",os.listdir("./datset/img_align_celeba/img_align_celeba")[0] )
    loader.applyRescale1(switch=True)
    normalScaledImg = loader.loadImage(path, applyTransform=True)
    print("normalScaled image max = ",normalScaledImg.max())
    print("normalScaled image min = ",normalScaledImg.min())

    print("Saving all kinds of images to default directory")
    loader.saveImage(img=unscaledImg,imgName="unscaledImg.jpg")
    print('Unscaled img saved')
    loader.saveImage(img=scaledImg,imgName="scaledImg.jpg")
    print('scaledImg img saved')
    loader.saveImage(img=normalScaledImg,imgName="normalScaledImg.jpg")
    print('normalScaledImg img saved')
    
