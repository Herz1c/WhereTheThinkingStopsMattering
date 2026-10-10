"""Run model-written Python functions against their unit tests, one stream per process.

Reads one JSON object per line on stdin, {"code": ..., "tests": ...}, and writes one
verdict per line, in order: {"passed": bool, "error": str or null}.

Each item runs in a forked child, so a solution that crashes, exits, recurses forever
or leaves global state behind cannot affect the next one. The child imports the code
as the module `solution` (the tests do `from solution import f`), executes the test
file, and calls every `test_*` function it defines. Time per item is capped.

Executes untrusted code: the caller provides the sandbox (see stopping/code_tasks.py).
"""

import json
import os
import signal
import sys
import types

TIME_LIMIT = 6      # seconds per item, wall clock


def run_item(code, tests):
    module = types.ModuleType("solution")
    exec(compile(code, "solution.py", "exec"), module.__dict__)
    sys.modules["solution"] = module
    # Some test files import from `solution`; others call the function directly, written to run
    # in the same file as the solution. Expose the solution's public names to both kinds.
    namespace = {k: v for k, v in module.__dict__.items() if not k.startswith("_")}
    namespace["__name__"] = "tests"
    exec(compile(tests, "test_solution.py", "exec"), namespace)
    found = [f for name, f in namespace.items() if name.startswith("test") and callable(f)]
    if not found:
        raise RuntimeError("no tests")
    for test in found:
        test()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        read, write = os.pipe()
        child = os.fork()
        if child == 0:
            os.close(read)
            signal.alarm(TIME_LIMIT)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            try:
                run_item(item["code"], item["tests"])
                message = "ok"
            except BaseException as error:     # noqa: BLE001 -- any failure is a verdict
                message = f"{type(error).__name__}: {str(error)[:200]}"
            os.write(write, message.encode()[:4000])
            os._exit(0)
        os.close(write)
        chunks = []
        while True:
            data = os.read(read, 4096)
            if not data:
                break
            chunks.append(data)
        os.close(read)
        _, status = os.waitpid(child, 0)
        message = b"".join(chunks).decode(errors="replace")
        if not message:
            message = "killed" if os.WIFSIGNALED(status) else "no result"
        print(json.dumps({"passed": message == "ok", "error": None if message == "ok" else message}), flush=True)


if __name__ == "__main__":
    main()
