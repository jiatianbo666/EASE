"""M7 指标冒烟 —— EM/F1 与 HotpotQA 官方口径逐项手算核对（纯确定性，无 API）。"""
import sys
import os

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ease.eval.metrics import normalize, token_f1, em_f1, answer_success


def main():
    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  {'✅' if cond else '❌'} {name}" + (f" | {detail}" if detail else ""))
        if not cond:
            ok = False

    print("[1] normalize（HotpotQA 官方）")
    check("小写", normalize("The Answer") == "answer")
    check("去冠词", normalize("the answer an apple") == "answer apple")
    check("去标点", normalize("Answer, please!") == "answer please")
    check("合并空白", normalize("a  b   c") == "b c")

    print("\n[2] token_f1 手算")
    # pred={a,b,c} gold={b,c,d}: common={b,c} prec=2/3 rec=2/3 f1=(4/3)/(4/3)=1.0? no:
    # f1 = 2*(2/3)*(2/3)/(2/3+2/3) = 2*(4/9)/(4/3)= (8/9)/(4/3)= (8/9)*(3/4)=2/3
    f1 = token_f1("a b c".split(), "b c d".split())
    check("f1={b,c}∩", abs(f1 - 2.0 / 3.0) < 1e-9, f"f1={f1:.4f}（手算 2/3）")
    check("无交集=0", token_f1("a".split(), "z".split()) == 0.0)
    check("空输入=0", token_f1([], ["a"]) == 0.0)
    check("完全一致=1", token_f1("the cat".split(), "the cat".split()) == 1.0)

    print("\n[3] em_f1 对 gold 列表取最优")
    e, f = em_f1("american", ["american", "USA"])
    check("EM 命中", e == 1.0)
    e2, f2 = em_f1("america", ["american", "USA"])
    check("F1 部分", abs(f2 - 1.0) > 0 and e2 == 0.0, f"e2={e2} f2={f2:.3f}")
    e3, _ = em_f1("", ["american"])
    check("空答案=0", e3 == 0.0)

    print("\n[4] answer_success 门控")
    check("EM>0 → 成功", answer_success("yes", "yes"))
    check("F1>=1.0 → 成功", answer_success("United States", "united states"))
    check("无关 → 失败", not answer_success("french", "american"))

    print("\n" + "=" * 50)
    print("✅ METRICS SMOKE PASSED" if ok else "❌ 存在失败项")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
