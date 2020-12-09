from emb_algos import example, glove_emb

class Embeddings():
    def __init__(self, algo):
        # Instantiate the class of the selected algorithm
        self.nlp_algo = algo()
        pass

    def load_description(self, text_path):
        self.text_path = text_path

    def preprocess(self, text):
        self.nlp_algo.preprocess(text)    

    def encode(self, text):
        self.nlp_algo.encode(text)

if(__name__ == "__main__"):
    # Initialize
    emb_algos = {'glove':glove_emb.GloveEmbeddings, "example":example.example}
    nlp = Embeddings(emb_algos['glove'])
    nlp.preprocess("hello")
    nlp.encode("test")