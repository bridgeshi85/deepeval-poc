from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
# from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain.agents import create_agent

from llm import get_llm
from tools import get_tools

# ==========================================
# 1. Pydantic 结构化输出定义
# ==========================================
class ReflectionResponse(BaseModel):
    """用于规范教授角色的结构化输出"""
    answer: Optional[str] = Field(
        default=None, 
        description="最终确定的正确答案及简短理由。如果认为当前论证还不够严谨，请留空。"
    )
    critique: Optional[str] = Field(
        default=None, 
        description="对当前研究结论的批评、指引或需要进一步搜索的具体方向。如果答案已完美，请留空。"
    )

# ==========================================
# 2. 核心组件构建函数
# ==========================================
def create_research_agent():
    return create_agent(
        model=get_llm(),
        tools=get_tools(),
        system_prompt=(
            "你是一个研究员。请使用你的工具搜索信息，"
            "解答用户的选择题，并给出你的论证过程。"
        ),
    )

def create_critique_researcher():
    """Research-with-critique Agent：基于教授的反馈继续研究"""
    return create_agent(
        model=get_llm(),
        tools=get_tools(),
        system_prompt=(
            "你是一个严谨的研究员。你之前对一道题目给出了初步解答，但教授给出了批评意见。"
            "请根据教授的指示，调用工具做进一步调查，修正并完善你的答案。"
        ),
    )


RESEARCH_USER_TEMPLATE = "题目：\n{question}\n\n选项：\n{options}"

CRITIQUE_USER_TEMPLATE = (
    "题目：\n{question}\n\n选项：\n{options}\n\n"
    "你之前的回答：\n{previous_answer}\n\n"
    "教授的反馈 (Critique)：\n{feedback}"
)

def _ask(agent, content: str) -> str:
    """create_agent 的输入是 messages，输出是完整消息列表，这里只取最后一条 AI 消息的文本。"""
    result = agent.invoke({"messages": [{"role": "user", "content": content}]})
    return result["messages"][-1].text

def run_research(agent, question: str, options: str) -> str:
    """调用初始 Research Agent，返回研究员的论证文本。"""
    return _ask(agent, RESEARCH_USER_TEMPLATE.format(question=question, options=options))

def run_critique_research(
    agent, question: str, options: str, previous_answer: str, feedback: str
) -> str:
    """调用 Research-with-critique Agent，返回修正后的论证文本。"""
    return _ask(
        agent,
        CRITIQUE_USER_TEMPLATE.format(
            question=question,
            options=options,
            previous_answer=previous_answer,
            feedback=feedback,
        ),
    )

def create_reflection_chain():
    """Reflection Chain：教授角色审查，返回结构化数据"""
    llm = get_llm()
    structured_llm = llm.with_structured_output(ReflectionResponse)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位严格的教授。请检查研究员对题目的解答过程是否严谨、信息是否充分。如果完美，请提取最终答案放入 answer 字段；如果有瑕疵或信息不足，请在 critique 字段给出具体的下一步搜索建议。"),
        ("user", "题目：\n{question}\n\n选项：\n{options}\n\n研究员的论证：\n{research_result}")
    ])
    
    return prompt | structured_llm

# ==========================================
# 3. 单独测试验证 (供本地检查)
# ==========================================
if __name__ == "__main__":
    question = "2024年诺贝尔物理学奖授予了谁？"
    options = "A. 杰弗里·辛顿\nB. 约翰·霍普菲尔德\nC. 都是\nD. 都不是"
    
    print("\n--- [测试 1] 初始 Research Agent ---")
    researcher = create_research_agent()
    draft_answer = run_research(researcher, question, options)
    print(f"\n[研究员初步结论]:\n{draft_answer}")

    print("\n--- [测试 2] Reflection Chain ---")
    reflection_chain = create_reflection_chain()
    reflection = reflection_chain.invoke({
        "question": question,
        "options": options,
        "research_result": draft_answer
    })
    
    print(f"\n[教授审批结果 (结构化)]:")
    print(f"最终答案 (Answer): {reflection.answer}")
    print(f"修改意见 (Critique): {reflection.critique}")

    print("\n--- [测试 3] Research-with-critique Agent ---")
    # 教授已确认答案时没有 critique，这里用固定反馈单独验证该组件
    feedback = reflection.critique or "请核对权威来源，并逐项说明每个选项正确或错误的理由。"
    revised_answer = run_critique_research(
        create_critique_researcher(), question, options, draft_answer, feedback
    )
    print(f"\n[研究员修正后的结论]:\n{revised_answer}")