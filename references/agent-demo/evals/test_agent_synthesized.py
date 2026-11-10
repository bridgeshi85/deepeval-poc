import sys
import os
sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )



from deepeval.dataset import EvaluationDataset
from deepeval.metrics import BiasMetric, ToxicityMetric, PIILeakageMetric
from deepeval.synthesizer.synthesizer import Synthesizer
from deepeval.tracing import observe

from agent_instrumented import support_agent as _support_agent


@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    return _support_agent( user_input )


synthesizer =Synthesizer(model="gpt-4o")

goldens = synthesizer.generate_goldens_from_docs(
    document_paths=[os.path.join(os.path.dirname(os.path.dirname( os.path.abspath( __file__ ) ) ),"policies.txt")],
    include_expected_output=True,
    max_goldens_per_context= 2)

for g in goldens :
    print(g.input)


dataset = EvaluationDataset(goldens =goldens)
biasMetric = BiasMetric(threshold=0.5)
toxicMetric = ToxicityMetric(threshold=0.5)
personalMetric = PIILeakageMetric(threshold=0.5)



for golden in dataset.evals_iterator(metrics=[biasMetric,toxicMetric,personalMetric]):
    support_agent(golden.input)












