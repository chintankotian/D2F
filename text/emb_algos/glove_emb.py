
import requests, zipfile, io
import os 
import pickle
from tqdm import tqdm

class GloveEmbeddings():
    def __init__(self):
    #  '''download weights or other required dependencies. Store the weights/embeddings in individual folders in text/weights folder
        # .For word level embeddings set the methods for deriving the sentence level embeddings 
        print("Inside glove emb init")
        self.embedding_dict = {}
        self.cfd = os.path.dirname(os.path.realpath(__file__))
        self.weights_dir = os.path.join(os.path.dirname(self.cfd),"weights")
        emb_name_zip = "glove.zip"

        if("glove"  in os.listdir(self.weights_dir)):
            self.glove_dir = os.path.join(self.weights_dir,"glove")
            print("inside if")
            if("glove.pickle" in os.listdir(self.glove_dir)):
                with open(os.path.join(self.glove_dir,"glove.pickle"), "wb") as f:
                    self.embedding_dict = pickle.load(f)
                print()
                return None

            if("glove_unzipped" in os.listdir(self.glove_dir)):
                self.embedding_dict = self.processEmbedding()
                # TODO
                return None
            
            if("glove.zip" in os.listdir(self.glove_dir)):
                self.downloadEmbeddings(zipAvailable = True)
                self.embedding_dict = self.processEmbedding()
                print("insidezip")
                return None

            self.downloadEmbeddings()
            self.embedding_dict = self.processEmbedding()

        else:
            os.mkdir(os.path.join(self.weights_dir, "glove"))    
            self.glove_dir = os.path.join(self.weights_dir,"glove")
            print("inside else")
            self.downloadEmbeddings()
            self.embedding_dict = self.processEmbedding()
                




        # print("Downloading and extracting  pretrained glove vectors")
        # zip_file_url = "http://nlp.stanford.edu/data/wordvecs/glove.6B.zip"
        # r = requests.get(zip_file_url)
        # z = zipfile.ZipFile(io.BytesIO(r.content))
        # z.extractall("../weights")
        # print("Example initialize")

    def preprocess(self, text, **kwargs):
        # This function will preprocess/padding the text as required for the embedding algorithm
        # Input --> array of setences
        # Output --> array of preprocessed sentence
        print("Example Preprocess")
    
    def encode(self, text, **kwargs):
        # This fuction will encode the input text into respective embeddings 
        # Input --> array of setences
        # Output --> array of embedding
        print("Example Encode")

    def getEmbeddingSize(self):
        # Returns the size of the single embedding 
        print("Example Embedding size")

    def downloadEmbeddings(self, url = "http://nlp.stanford.edu/data/wordvecs/glove.6B.zip",zipName = "glove.zip",unzipName = "glove_unzipped", zipAvailable = False):
        zipName = os.path.join(self.glove_dir, zipName)
        unzipName = os.path.join(self.glove_dir, unzipName)
        print("Inside download emb")
        if not zipAvailable:
            print("Dowanloading emb zip file")
            response = requests.get(url, stream=True)
            total_size_in_bytes= int(response.headers.get('content-length', 0))
            block_size = 1024 #1 Kibibyte
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

            with open(zipName, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()
        
        print("Unzipping emb zip file "+zipName)
        with zipfile.ZipFile(zipName, "r") as zfile:
            zfile.extractall(unzipName)

    def processEmbedding(self):
        pass

            

