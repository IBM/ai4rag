from unitxt.eval_utils import evaluate
from statistics import fmean
import dspy
from ai4rag.rag.chunking.chunk import AI4RAGChunk


def unitxt_metrics(example, prediction):
    """Given an example and model's prediction calculates:
    - answer correctness
    - faifhtfulness
    using the `unitxt.eval_utils.evaluate` function
    """
    eval_data = {
        "question": example.question,
        "answer": prediction.answer_grounded_in_contexts,
        "contexts": [chunk.text for chunk in example.contexts],
        "ground_truths": example.correct_answer,
    }
    scores, ci = evaluate(
        [eval_data],
        metric_names=[
            "metrics.rag.external_rag.answer_correctness",
            "metrics.rag.external_rag.faithfulness",
        ],
        compute_conf_intervals=True,
    )
    return scores, ci


def overall_score_per_question(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """For a given question calculates arithmetic mean from the following unitxt metrics:
    - answer_correctness
    - faithfulness
    - context_correctness
    """
    scores, ci = unitxt_metrics(example, prediction)

    for external_rag_metrics in (
        dict(filter(lambda items: "metrics.rag.external_rag" in items[0], per_question_score.items()))
        for per_question_score in scores
    ):
        return round(fmean(external_rag_metrics.values()), 4)


class RAGPrompt(dspy.Signature):

    question: str = dspy.InputField()
    contexts: list[AI4RAGChunk] = dspy.InputField()
    answer_grounded_in_contexts: str = dspy.OutputField()
