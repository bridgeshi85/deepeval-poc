# ==========================================
# 1. 定义图的全局状态 (State)
from typing import TypedDict, Optional, Literal
from agents import (
    create_research_agent,
    create_critique_researcher,
    create_reflection_chain
)
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

MAX_STEPS = 3  # 研究轮数上限，防止教授一直不满意导致无法终止

class ReflectionAgentState(TypedDict):
    """
    TypedDict 默认行为是覆盖更新（除非使用 Annotated 指定 reducer）。
    各个节点只需返回需要更新的字段字典即可。
    """
    question: str
    options: str
    current_answer: Optional[str]  # 存放研究员当前的答案草稿，或最终确认的答案
    feedback: Optional[str]        # 存放教授给出的修改意见
    steps: int                     # 记录循环迭代步数，防止死循环
    

# ==========================================
# 2. 预加载
# ==========================================
initial_researcher = create_research_agent()
critique_researcher = create_critique_researcher()
reflection_chain = create_reflection_chain()

# ==========================================
# 3. 定义节点 (Nodes)
# ==========================================
def research_start(state: ReflectionAgentState):
    """首次研究节点：读取题目，调用工具，输出初代草稿"""
    print(f"\n--- [Node: research_start] 开始首次研究 ---")
    result = initial_researcher.invoke({
        "question": state["question"],
        "options": state["options"]
    })
    print(f"[research_start] 初代草稿:\n{result}")
    # 返回增量更新：初始化步数为1，写入初始答案
    return {
        "current_answer": result,
        "steps": 1
    }


def research(state: ReflectionAgentState):
    """后续研究节点：基于反馈（critique）修正答案"""
    print(f"\n--- [Node: research] 收到反馈，继续研究 (当前第 {state.get('steps', 0)} 轮) ---")
    result = critique_researcher.invoke({
        "question": state["question"],
        "options": state["options"],
        "previous_answer": state["current_answer"],
        "feedback": state["feedback"]
    })
    
    return {
        "current_answer": result["output"],
        "steps": state.get("steps", 0) + 1
    }


def reflect(state: ReflectionAgentState) -> Command[Literal["research", "__end__"]]:
    """反思审查节点：教授角色校验当前答案，并决定结束还是继续研究"""
    print(f"\n--- [Node: reflect] 教授审查中 ---")
    reflection = reflection_chain.invoke({
        "question": state["question"],
        "options": state["options"],
        "research_result": state["current_answer"]
    })

    answer, critique = reflection.answer, reflection.critique

    # 两者都有或都没有，都视为教授没有确认，继续研究。
    approved = bool(answer) and not critique
    print(f"[reflect] 教授审查结果: answer={answer}, critique={critique}, approved={approved}")
    # 如果教授approve或者已经超过最大尝试步数，goto END；否则继续研究
    goto = END if approved or state.get("steps", 0) >= MAX_STEPS else "research"

    return Command(
        update={
            # 教授没有返回 answer 时保留研究员的草稿，避免被 None 覆盖
            "current_answer": answer or state["current_answer"],
            "feedback": critique,
        },
        goto=goto,
    )

# ==========================================
# 4. 组装并编译图 (Wiring)
# ==========================================
workflow = StateGraph(ReflectionAgentState)
workflow.add_node("research_start", research_start)
workflow.add_node("research", research)
workflow.add_node("reflect", reflect)

# reflect 出口由其返回的 Command(goto=...) 决定，这里只声明确定的边
workflow.add_edge(START, "research_start")
workflow.add_edge("research_start", "reflect")
workflow.add_edge("research", "reflect")

app = workflow.compile()

# ==========================================
# 5. 测试运行入口
# ==========================================
if __name__ == "__main__":
    inputs = {
        "question": "2024年诺贝尔物理学奖授予了谁？",
        "options": "A. 杰弗里·辛顿\nB. 约翰·霍普菲尔德\nC. 都是\nD. 都不是"
    }
    
    # 运行图并打印最终状态
    final_state = app.invoke(inputs)
    
    print("\n\n=== 最终结果 ===")
    print(final_state["current_answer"])