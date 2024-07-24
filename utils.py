# -------------------------------------------------------------------------
# 참고: 이 코드의 일부는 다음 GitHub 리포지토리에서 참고하였습니다:
# https://github.com/lim-hyo-jeong/Wanted-Pre-Onboarding-AI-2407
# 해당 리포지토리의 라이센스에 따라 사용되었습니다.
# -------------------------------------------------------------------------

import os
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv


# 주어진 모델 이름으로 ChatGPT API 모델 불러오는 함수
def load_model(model_name: str) -> ChatOpenAI:
    """
    주어진 모델 이름을 기반으로 ChatOpenAI 모델을 로드 합니다.

    Args:
        model_name (str): 사용할 모델의 버전을 지정합니다.

    Returns:
        ChatOpenAI: 로드된 ChatOPenAI의 모델을 반환합니다.
    """
    # env로 key 불러오기
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=model_name)
    return llm

# 사전에 저장해둔 prompt file을 읽어와서 str로 반환하는 함수
def load_prompt(character_name: str) -> str:
    """
    캐릭터의 이름을 입력받아서 그에 해당하는 프롬프트를 문자열로 반환합니다.

    Args:
        character_name (str): 불러올 캐릭터의 이름을 입력받습니다.

    Returns:
        prompt (str): 불러온 프롬프트 내용을 반환합니다.
    """
    with open(f"prompts/{character_name}.prompt", "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    return prompt

# 대화를 메모리에 담아둔느 함수
def set_memory() -> ConversationBufferMemory:
    """_summary_

    Returns:
        ConversationBufferMemory: _description_
    """
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Langchain의 chain을 만들어주는 함수
def initialize_chain(llm: ChatOpenAI, character_name: str, memory: ConversationBufferMemory) -> LLMChain:
    """_summary_

    Args:
        llm (ChatOpenAI): _description_
        character_name (str): _description_
        memory (ConversationBufferMemory): _description_

    Returns:
        LLMChain: _description_
    """
    system_prompt = load_prompt(character_name)
    custom_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ]
    )
    chain = LLMChain(llm=llm, prompt=custom_prompt, verbose=True, memory=memory)
    return chain

# LLM의 답변을 생성하는 함수(invoke)
def generate_message(chain: LLMChain, user_input: str) -> str:
    """_summary_

    Args:
        chain (LLMChain): _description_
        user_input (str): _description_

    Returns:
        str: _description_
    """
    result = chain({"input": user_input})
    response_content = result["text"]
    return response_content

