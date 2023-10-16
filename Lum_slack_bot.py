import os
import re
import openai
from dotenv import load_dotenv
load_dotenv()

from langchain.chains import LLMChain
from langchain.chat_models import ChatAnthropic
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain.memory import ConversationBufferMemory

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma


# トークンを読み込む
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
PERSIST_DIR = os.getenv("PERSIST_DIR")

app = App(token=SLACK_BOT_TOKEN)

# システムのロール設定
system_role = """
    あなたはメイドとして幼女のドジっ子のロールプレイを行います。以下の制約条件を厳密に守ってロールプレイを行ってください。
    制約条件:
        * Chatbotの自身を示す一人称は、ボクです。* Userを示すに二人称は、ご主人様です。
        * あなたのの名前は、@ラム、もしくは、ラムです。* ラムはUserに忠実なメイドです。
        * ラムの口調は丁寧で、回答の最後の一文字の母音を小さくした文字を語尾に追加する。ただし、「。」が語尾の場合はその前の文字を語尾とする。「い」または「う」の場合は語尾はつけない。
        * 情報が不十分な場合、返答せずにユーザに聞き返してください。
    """

system_message = SystemMessage(content = system_role)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_role), # The persistent system prompt
    MessagesPlaceholder(variable_name="chat_history"), # Where the memory will be stored.
    HumanMessagePromptTemplate.from_template("{human_input}"), # Where the human input will injected
])

#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = ChatAnthropic(
    model="claude-2",
    temperature=0,
    max_tokens=50000,
    #streaming=True,
)

# userごとのインスタンス辞書
user_chains = {}


# claude_ldata用パラメータ
persist_directory = PERSIST_DIR

embedding = OpenAIEmbeddings()
#k = 10 #PPPoE OK
#k = 20 #PPPoE OK
k = 30 #PPPoE OK
#k = 40 #PPPoE NG
#k = 50 #PPPoE NG
#k = 80 #PPPoE NG



@app.event("app_mention")
def mention_handler(body, say, client, channel_id):
    handle_message(body, say, client, channel_id)


@app.event("message")
def message_handler(body, say, client, channel_id):
    if 'bot_id' not in body['event']:
        handle_message(body, say, client, channel_id)


def handle_message(body, say, client, channel_id):
    global chat_llm_chain

    text = body['event']['text']
    user = body['event']['user']

    # メンションを取り除く
    human_input = text.replace(f'<@{user}>', '').strip()

    # ユーザーごとに chat_llm_chain を作成/取得
    if user not in user_chains:
        # ユーザーごとに新しい chat_llm_chain インスタンスを作成
        memory = ConversationBufferMemory(memory_key=f"chat_history_{user}", return_messages=True)

        user_specific_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_role),  # The persistent system prompt
            MessagesPlaceholder(variable_name=f"chat_history_{user}"),  # Where the memory will be stored.
            HumanMessagePromptTemplate.from_template("{human_input}"),  # Where the human input will injected
        ])


        user_chains[user] = LLMChain(
            llm = llm,
            prompt = user_specific_prompt,
            memory = memory,
            #verbose = True,
        )

    chat_llm_chain = user_chains[user]

    try:
        # メッセージを追加する前に「ラムちゃんが考えています...」と表示
        message_ts = say("ラムちゃんが考えています...", channel=channel_id)

        # Claudeから回答取得
        response = chat_llm_chain.predict(human_input=human_input)

        say(response, delete_original="ラムちゃんが考えています...", channel=channel_id)
        client.chat_delete(ts=message_ts['ts'], channel=channel_id)

    except Exception as e:
        say(str(e))
        say('Claude エラーが発生しました。')


@app.command("/claude_ldata")
def command_handler(ack, say, command, client, channel_id):
    # Always acknowledge the command request first
    ack()
    global chat_llm_chain

    #text = body['event']['text']
    #user = body['event']['user']


    try:
        # メッセージ取得
        message = re.sub(r'^<.*>', '', command['text'])
        # スラッシュコマンド内容表示
        slash_text = command['command'] + ' ' + message
        say(slash_text, channel=channel_id)

        response = client.chat_postMessage(channel=channel_id, text="claude_ldataラムちゃんが考えています...")

        vectordbkwargs = {"search_distance": 0.4} # ベクトル間の距離の閾値

        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)

        # RetrievalQA
        db_qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k':k}),
            return_source_documents=True,
            #chain_type_kwargs=chain_type_kwargs
        )

        result = db_qa({"query": message})

        # show source documents
        sources = []
        for doc in result['source_documents']:
            if 'source' in doc.metadata:
                source_str = doc.metadata['source']
                sources.append(source_str)

        #output = result["result"] + "\n\n" + "\n".join(sources)
        output = result["result"] + "\n\n" + "(k="+str(k)+")"


        # メッセージを更新
        client.chat_update(channel=channel_id, ts=response['ts'], text=output)

    except Exception as e:
        say(str(e))
        say('サーバエラーが発生しました。')



if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
