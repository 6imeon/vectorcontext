"""Minimal persistent Python REPL sandbox: exec code with a shared namespace, capture stdout, echo last expression."""
from __future__ import annotations
import ast, io, contextlib, traceback, threading, re, json, collections, sys

class Sandbox:
    def __init__(self, namespace: dict, output_limit: int = 4000, timeout: int = 10):
        self.ns = {"re": re, "json": json, "collections": collections, "Counter": collections.Counter, "__builtins__": __builtins__}
        self.ns.update(namespace)
        self.output_limit = output_limit; self.timeout = timeout
    def run(self, code: str) -> str:
        buf = io.StringIO()
        def target():
            try:
                tree = ast.parse(code)
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last = ast.Expression(tree.body.pop().value)
                    if tree.body: exec(compile(tree, "<repl>", "exec"), self.ns)
                    val = eval(compile(last, "<repl>", "eval"), self.ns)
                    if val is not None: print(repr(val), file=buf)
                else:
                    exec(compile(tree, "<repl>", "exec"), self.ns)
            except Exception:
                print("".join(traceback.format_exception_only(*sys.exc_info()[:2])).strip(), file=buf)
        # redirect print() inside sandbox code to buf via a namespace-level print
        self.ns["print"] = lambda *a, **k: print(*a, **{**k, "file": buf})
        t = threading.Thread(target=target, daemon=True); t.start(); t.join(self.timeout)
        if t.is_alive(): print(f"[TimeoutError: code exceeded {self.timeout}s]", file=buf)
        out = buf.getvalue()
        if not out.strip(): out = "(no output)"
        if len(out) > self.output_limit:
            out = out[: self.output_limit] + f"\n...[truncated, {len(out)-self.output_limit} more chars; narrow your query]"
        return out
