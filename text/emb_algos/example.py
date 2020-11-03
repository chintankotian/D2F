# A template for making other nlp algo code

class example():
    def __init__(self):
        # download weights or other required dependencies. Store the weights/embeddings in individual folders in text/weights folder
        # for word level embeddings set the methods for deriving the sentence level embeddings 
        print("Example initialize")

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
