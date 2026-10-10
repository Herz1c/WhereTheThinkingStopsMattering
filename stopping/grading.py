"""One grading entry point for every task type, in batches.

    from stopping.grading import grade
    grade([(row, completion_text), ...])  ->  [(strict, knows), ...]

`strict` is what the evaluation would accept; `knows` is whether the answer is right
however it is formatted. For math and multiple choice these are utils/train/rewards.py
and the lenient grader of analysis/forced_stop_sweep.py. For code both mean "passes the
unit tests", strict additionally requiring exactly one fenced code block
(stopping/code_tasks.py). For tool calls both mean "the right call, or rightly no call",
strict additionally requiring the bare call list (stopping/tool_tasks.py). Code is graded in
one parallel sandboxed batch, which is why this takes a list instead of one answer at a time.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.forced_stop_sweep import knows  # noqa: E402
from utils.train.rewards import is_task_correct  # noqa: E402


def grade(pairs):
    results = [None] * len(pairs)
    code = [(k, row, text) for k, (row, text) in enumerate(pairs) if row["task"] == "code"]
    for k, (row, text) in enumerate(pairs):
        if row["task"] == "tool":
            from stopping.tool_tasks import grade_tool
            results[k] = grade_tool(row, text)
        elif row["task"] != "code":
            results[k] = (is_task_correct(text, row["answer"], row["task"]),
                          knows(text, row["answer"], row["task"]))
    if code:
        from stopping.code_tasks import grade_code
        for (k, _, _), verdict in zip(code, grade_code([(row, text) for _, row, text in code])):
            results[k] = verdict
    return results
