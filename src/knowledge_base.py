from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from config import EMBEDDING_MODEL

class KnowledgeBase(object):
    def __init__(self, embedding_model):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
    def load_docs(self, path):
        loader = DirectoryLoader(path, glob='**/*.md', show_progress=True, recursive=True)
        self.docs = loader.load()
    
    def get_index_from_doc(self):
        self.db = FAISS.from_documents(self.docs, self.embeddings)
        return self.db
    
    def save_index(self, dest, index_name):
        self.db.save_local(dest, index_name)
    
    def get_index_from_local(self, dest, index_name):
        self.db = FAISS.load_local(dest, self.embeddings, index_name)
        
    def similarity_search(self, query, k=3):
        result = self.db.similarity_search(query, k=k)
        return result
        
if __name__ == '__main__':
    knowledge_base = KnowledgeBase(EMBEDDING_MODEL)
    # knowledge_base.load_docs()
    # knowledge_base.get_index_from_doc()
    # knowledge_base.save_index('./index', 'my_workspace')
    knowledge_base.get_index_from_local('./index', 'my_workspace')
    print(knowledge_base.similarity_search('宁波大学的纳税人识别号是什么'))