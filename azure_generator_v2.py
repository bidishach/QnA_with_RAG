from openai import AzureOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.docstore.document import Document

class AzureGenerator():
    def __init__(self, docs: List[Document]) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = FAISS.from_documents(docs, self.embeddings)
        
        self.client = AzureOpenAI( api_key= 'your api key',
                            api_version="your api version",
                            azure_endpoint = "your api end point"
        )
        self.count_1 = 0
        self.count_k = 0
        self.retrieved_docs = None       
        
    def __get_retrieval_score(self, docs: List[Document], i,countk, count1):
        if i in [ docs[i].metadata['row'] for i in range(len(docs)) ]:
            countk += 1
            
        if i == docs[0].metadata['row']:
            count1 += 1
        return countk, count1

    def __fetch_k(self, query, i=None, k=3):
               
        self.retrieved_docs  = self.db.similarity_search(query, k)  
        if i is not None:      
            self.count_k, self.count_1 = self.__get_retrieval_score(self.retrieved_docs , i,self.count_k, self.count_1)   

    def get_row_number(self, query, i):
        self.__fetch_k(query,i)
        return [doc.metadata['row'] for doc in self.retrieved_docs]

    def __create_prompt(self, query, k=3):
        
        self.__fetch_k(query)
        context = '\n'.join([self.retrieved_docs[i].page_content for i in range(k)]) # taking the top most

        header = '''
                        Create a concise and informative answer (no more than 50 words) for a given question
                        based solely on the given contexts. You must only use information from the given contexts.
                        Use an unbiased and journalistic tone. Do not repeat text.
                        If the documents do not contain the answer to the question, say that ‘answering is not possible given the
                        available information.’ If the question contains profanity, say that 'Profanity detected in question.
                        Please rephrase and ask again.'\n
                '''
        return header + context + "\n\n" + query + "\n"   

    def generate(self, query, k=3):
        prompt = self.__create_prompt(query, k)

        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": prompt},
            {"role": "user", "content": query }
        ]

        response = self.client.chat.completions.create(
                                model="test",
                                messages=messages,
                                temperature=0,
                                max_tokens=1000,
                                top_p=1,
                                frequency_penalty=0,
                                presence_penalty=0,
                                stop = [' END']
                                )

        
        return response.choices[0].message.content