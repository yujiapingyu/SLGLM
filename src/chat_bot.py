import logging

from transformers import AutoModel, AutoTokenizer
from knowledge_base import KnowledgeBase


class ChatBot(object):
    def __init__(self, model_name) -> None:
        logging.info('Loading model [{}]...'.format(model_name))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()
        logging.info('Loading model success [{}].'.format(model_name))
        self.history = []
        
    def stream_chat(self, new_chat=True, max_history=20, with_knowledge_base=False):
        if new_chat:
            self.clean_history()
            
        while True:
            input_text = input(">>> ")
            if input_text == 'quit' or input_text == 'exit' or input_text == 'q':
                print('Bye!')
                break
            
            if input_text == '':
                continue
            
            
            if with_knowledge_base and self.knowledge_base:
                template = '我将给你一些文档内容以及来源，请你阅读并回答我的问题，你可以对文档进行总结，也可以加入自己已有的知识。答案不需要包含没用的话。\n'
                similar_docs = self.knowledge_base.similarity_search(input_text, k=10)
                index = 1
                for doc in similar_docs:
                    template += '文档{index}: {content} \n'.format(content=doc.page_content, index=index)
                    index += 1

                input_text = '{} 我的问题是：{}'.format(template, input_text)
                
            print(input_text)
            
            last_resp = ''
            current_history = None
            history = self.history[-max_history:]
            if max_history == 0:
                history = []
            print('Medical小🐷手: ', end='')
            for current_resp, current_history in self.model.stream_chat(self.tokenizer, input_text, history=history):
                print(current_resp[len(last_resp):], end='', flush=True)
                last_resp = current_resp
            print()
            self.history = current_history
            
    def clean_history(self):
        self.history = []
        
    def set_history(self, history):
        self.history = history
        
    def set_knowledge_base(self, knowledge_base):
        self.knowledge_base = knowledge_base
    
    
if __name__ == '__main__':
    from config import MODEL_NAME, EMBEDDING_MODEL
    bot = ChatBot(MODEL_NAME)
    knowledge_base = KnowledgeBase(EMBEDDING_MODEL)
    knowledge_base.get_index_from_local('./index', 'medisian')
    bot.set_knowledge_base(knowledge_base)
    bot.stream_chat(with_knowledge_base=True, max_history=0)

    

