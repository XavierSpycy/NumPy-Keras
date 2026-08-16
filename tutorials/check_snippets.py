#!/usr/bin/env python3
"""tutorials/check_snippets.py -- 校验 md 代码块与 code/*.py 的一致性

规则（写文章时的固定约定）：
1. 每篇 md 必须包含**一个**与同编号 code/*.py 逐字节一致的
   ```python 代码块（忽略末尾换行差异）——读者整段复制即可运行。
2. 首行为 `# excerpt: <目标>` 的代码块是节选：去掉该首行后，剩余内容
   必须作为连续子串（含缩进）出现在目标文件中。
   <目标> 以 `numpy_keras/`、`tests/` 开头或以 `.py` 结尾时指向
   仓库内该文件（用于引用库源码片段）；其余情况指向本篇的 code 文件。

用法：python tutorials/check_snippets.py [可选的文章编号前缀，如 00]
退出码：0 = 全部通过；1 = 存在不一致或缺失。
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # tutorials/
REPO = ROOT.parent                              # 仓库根
CODE_DIR = ROOT / "code"


def python_blocks(md_path: Path):
    text = md_path.read_text(encoding="utf-8")
    return re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)


def excerpt_source(target: str, code_path: Path, failures: list, block_no: int):
    if (target.startswith("numpy_keras/") or target.startswith("tests/")
            or target.endswith(".py")):
        source_path = REPO / target
        if not source_path.exists():
            failures.append(f"  block {block_no} (excerpt): 目标文件不存在: {target}")
            return None
        return source_path.read_text(encoding="utf-8").rstrip("\n")
    return code_path.read_text(encoding="utf-8").rstrip("\n")


def check_article(md_path: Path, code_path: Path):
    code = code_path.read_text(encoding="utf-8").rstrip("\n")
    blocks = python_blocks(md_path)
    failures = []
    n_full = 0
    for i, block in enumerate(blocks, 1):
        lines = block.split("\n")
        if lines[0].startswith("# excerpt:"):
            target = lines[0][len("# excerpt:"):].strip()
            source = excerpt_source(target, code_path, failures, i)
            if source is None:
                continue
            fragment = "\n".join(lines[1:]).rstrip("\n")
            if fragment not in source:
                failures.append(f"  block {i} (excerpt): 节选不是 {target or code_path.name} 的连续子串")
        else:
            n_full += 1
            if block.rstrip("\n") != code:
                failures.append(f"  block {i}: 与 {code_path.name} 不一致")
    if n_full == 0:
        failures.append("  缺少完整脚本代码块（需有一个 ```python 块与 code 文件逐字节一致）")
    return failures


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    n_ok, n_fail = 0, 0
    for code_path in sorted(CODE_DIR.glob(f"{prefix}*.py")):
        md_path = ROOT / f"{code_path.stem}.md"
        if not md_path.exists():
            print(f"[skip] {code_path.name}: 尚未找到 {md_path.name}")
            continue
        failures = check_article(md_path, code_path)
        if failures:
            n_fail += 1
            print(f"[FAIL] {md_path.name}")
            for f in failures:
                print(f)
        else:
            n_ok += 1
            print(f"[ ok ] {md_path.name} ({len(python_blocks(md_path))} 个 python 代码块)")
    print(f"\n{n_ok} 篇通过, {n_fail} 篇不一致")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
