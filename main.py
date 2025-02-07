from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from os import listdir
import json
import pandas as pd


df = pd.read_csv('math-project\external_df.csv')
df= df.loc[df['stage']=='test'][0:250]
df.drop(['source','stage'],axis=1,inplace=True)
print(df['problem'])

template = """
Answer the following math question and only provide the answer, without explaination.
question:{question}
Answer:
"""

model = OllamaLLM(model="llama3.1")
prompt = ChatPromptTemplate.from_template(template)
chain= prompt | model
def handle_conversation():
    context = ""
    print(len(df))
    for i in range(7500,len(df)+7500):
        user_input = df['problem'][i]
        result = chain.invoke({"question":user_input})
        context += f"\nUser:{user_input}\nAI:{result}"
        print('finish question',i)
    with open(r'math-project\try.txt','w') as f:
        f.writelines(context)
    f.close()
def handle_data():
    with open(r'math-project\try.txt','r') as f:
        data = f.readlines()
    f.close()
    user_data = []
    ai_data = []
    for i in data:
        if i.startswith('User'):
            user_data.append(i)
        elif i.startswith('AI'):
            ai_data.append(i)
    lst = []
    for i in range(len(user_data)):
        lst.append({'question':user_data[i],'answer':ai_data[i]})
    with open(r'math-project\aisolve.json','w') as f:
        json.dump(lst,f,indent=2) 
if __name__ == '__main__':
    handle_conversation()
    handle_data()

  
    



