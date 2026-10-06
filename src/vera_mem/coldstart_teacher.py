"""Fixed teacher-context interventions; common questions never change.

Only gold_annotated appends the answer label. It is an explicitly stronger
supervision condition, not a raw-context teacher or a source-only extraction.
No strategy selects examples or changes the writer/student observation.
"""
from __future__ import annotations

from .metrics import normalize_answer

STRATEGIES = ("baseline", "quote_instruction", "gold_annotated")
QUOTE_INSTRUCTION = (
    "Extraction instruction for the current question: Locate the exact anchor quoted in the question "
    "in the note above. Start immediately after the final character of that anchor, skip whitespace, "
    "and copy exactly the next three whitespace-delimited words in their original order and capitalization. "
    "Copy the literal source text; do not paraphrase, infer, complete a phrase, or include words from inside "
    "the anchor. Output only those three words separated by single spaces; no quotation marks, labels, or explanation."
)
GOLD_INSTRUCTION = (
    "The following requested_span is an explicit answer-location annotation for the current question. "
    "It identifies its three-word answer. Copy only the contents exactly in their original order and "
    "capitalization. Output these three words only; no label, tags, or explanation."
)


def question_exposes_answer(question,answer):
    """Whole normalized phrase only; articles stay and partial words do not match."""
    if not isinstance(question,str) or not isinstance(answer,str) or not normalize_answer(answer):
        raise ValueError("Question/answer must be text with a nonempty normalized answer")
    return (" "+normalize_answer(answer)+" ") in (" "+normalize_answer(question)+" ")


def strategy_metadata(strategy):
    if strategy not in STRATEGIES:
        raise ValueError("Unknown teacher strategy: "+str(strategy))
    return dict(strategy=strategy,question_unchanged=True,source_context_preserved=True,
        extra_gold_localization=strategy=="gold_annotated",
        source_only=strategy!="gold_annotated",
        interpretation=("Explicit answer-label localization supervision; not an original raw-context teacher."
            if strategy=="gold_annotated" else "Original source context and question, with no appended answer label."))


def make_teacher_context(record, context, answer, strategy):
    """Return a deterministic context; baseline/quote do not inspect the label.

    ``record`` is accepted for the training call contract but is deliberately not
    consulted: neither hidden metadata nor another world's label can enter a
    source-only prompt. The original context is an unchanged prefix of output.
    """
    strategy_metadata(strategy)
    if not isinstance(context,str) or not context:
        raise ValueError("Teacher context must be nonempty text")
    if strategy=="baseline": return context
    if strategy=="quote_instruction": return context+"\n\n"+QUOTE_INSTRUCTION
    if (not isinstance(answer,str) or len(answer.split())!=3 or " ".join(answer.split())!=answer
            or any(c in answer for c in "<>") or answer not in context):
        raise ValueError("Gold annotation needs an exact three-word literal answer already present in its context")
    if "<requested_span>" in context or "</requested_span>" in context:
        raise ValueError("Source already contains reserved requested_span annotation")
    return context+"\n\n"+GOLD_INSTRUCTION+"\n<requested_span>"+answer+"</requested_span>"


def make_teacher_question(record, question, strategy):
    """All conditions use the identical original question, including gold."""
    strategy_metadata(strategy)
    if not isinstance(question,str) or not question:
        raise ValueError("Teacher question must be nonempty text")
    return question
