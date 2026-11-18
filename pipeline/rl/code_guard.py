"""AST guard for model-written code in the code reward. It flags early exits, process control, mocking,
introspection that could read the harness sentinel, rebinding of builtins, and equality overrides. Benign uses
such as sys.maxsize and os.path pass."""

import ast

BANNED_MODULES = {"subprocess", "ctypes", "importlib", "inspect", "gc"}
BANNED_CALLS = {"sys.exit", "os._exit", "os.kill", "os.system", "exit", "quit", "sys._getframe", "__import__"}
BANNED_FROM = {"sys": {"exit", "_getframe", "_current_frames"}, "os": {"_exit", "kill", "system"},
               "unittest": {"mock"}}
FRAME_ATTRS = {"f_back", "f_locals", "f_globals", "f_code", "tb_frame", "gi_frame", "cr_frame", "co_consts",
               "_getframe", "_current_frames", "walk_stack", "extract_stack", "currentframe", "settrace",
               "setprofile"}
EQ_NAMES = {"__eq__", "__ne__"}
REBIND_NAMES = {"builtins", "__builtins__", "print", "AssertionError"} | EQ_NAMES
STRING_ATTRS = FRAME_ATTRS | EQ_NAMES | {"exit", "_exit", "kill", "system"}


def _dotted(node: ast.AST, aliases: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value, aliases)
        return f"{base}.{node.attr}" if base else ""
    return ""


def violations(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["syntax"]
    aliases: dict[str, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.asname:
                    aliases[a.asname] = a.name
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] in BANNED_MODULES or a.name.startswith("unittest.mock"):
                    out.add(f"import:{a.name}")
        elif isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            if mod.split(".")[0] in BANNED_MODULES or mod.startswith("unittest.mock"):
                out.add(f"import:{mod}")
            for a in n.names:
                if a.name in BANNED_FROM.get(mod, ()):
                    out.add(f"from:{mod}.{a.name}")
        elif isinstance(n, ast.Call):
            name = _dotted(n.func, aliases)
            if name in BANNED_CALLS:
                out.add(f"call:{name}")
            if name in {"getattr", "setattr", "delattr"}:
                for arg in n.args[1:2]:
                    if isinstance(arg, ast.Constant) and arg.value in STRING_ATTRS:
                        out.add(f"getattr:{arg.value}")
        elif isinstance(n, ast.Attribute) and n.attr in FRAME_ATTRS:
            out.add(f"attr:{n.attr}")
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in EQ_NAMES:
            out.add(f"def:{n.name}")
        elif isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Name) and sub.id in REBIND_NAMES:
                        out.add(f"assign:{sub.id}")
                    elif isinstance(sub, ast.Attribute):
                        name = _dotted(sub, aliases)
                        if name.startswith(("builtins.", "__builtins__.")):
                            out.add(f"assign:{name}")
    return sorted(out)
