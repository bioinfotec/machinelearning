"""
matplotlib 通用配置
====================
为仓库内所有学习脚本提供统一的中文字体支持,避免每个脚本
重复书写字体配置代码。

背景
----
matplotlib 默认字体 DejaVu Sans 不含 CJK 字形,若不显式配置,
图表中的中文会全部渲染成方框 □□□,并抛出大量
    UserWarning: Glyph 36845 (\\N{CJK UNIFIED IDEOGRAPH-8FED})
    missing from font(s) DejaVu Sans
警告。

用法
----
在任意绘图脚本的导入区加入:

    from plot_utils import setup_chinese_font
    setup_chinese_font()

函数会挑选系统中第一个实际已安装的中文字体,并顺带修复
负号(U+2212)显示为方框的问题。
"""
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 常见中文字体候选,按优先级排列,覆盖 Windows / macOS / Linux
CHINESE_FONT_CANDIDATES = [
    'Microsoft YaHei', 'SimHei',                     # Windows
    'PingFang SC', 'Hiragino Sans GB', 'Heiti SC',   # macOS
    'Noto Sans CJK SC', 'Source Han Sans SC',        # Linux 桌面
    'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',      # Linux 国产发行版
    'Droid Sans Fallback',                           # 其他 / 容器环境
]

# 中文字体缺失时的兜底字体(只提供拉丁字形,保证不会整体报错)
FALLBACK_FONT = 'DejaVu Sans'


def get_available_chinese_fonts():
    """返回当前系统中已安装的中文字体列表(按候选优先级排序)"""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return [name for name in CHINESE_FONT_CANDIDATES if name in installed]


def setup_chinese_font(verbose=True):
    """配置 matplotlib 使用中文字体,避免中文显示为方框。

    该函数幂等,可安全重复调用。

    参数
    ----
    verbose : bool
        是否打印配置结果,默认 True。

    返回
    ----
    str or None
        实际选中的中文字体名;若系统未安装任何候选中文字体则返回 None。
    """
    installed = {f.name for f in font_manager.fontManager.ttflist}

    selected = next((name for name in CHINESE_FONT_CANDIDATES if name in installed), None)

    if selected is not None:
        # 把选中的中文字体放在首位,DejaVu Sans 作为兜底,
        # 保证 × → 等非中文字符始终有字体可用
        plt.rcParams['font.sans-serif'] = [selected, FALLBACK_FONT]
    else:
        plt.rcParams['font.sans-serif'] = [FALLBACK_FONT]

    # matplotlib 默认用 U+2212(真正的减号 −)绘制负数,
    # 部分中文字体没有该字形,会导致坐标轴负数显示成方框。
    # 关闭后改用 ASCII 连字符 '-',规避此问题。
    plt.rcParams['axes.unicode_minus'] = False

    if verbose:
        if selected is not None:
            print(f'[字体] 中文字体已设置: {selected}')
        else:
            print('[字体] [警告] 未找到可用的中文字体,图表中文可能显示为方框。')
            print(f'[字体] 可尝试安装以下字体之一: {", ".join(CHINESE_FONT_CANDIDATES[:4])}')

    return selected


if __name__ == '__main__':
    # 自检:直接运行本文件可查看当前系统的中文字体情况
    print('系统已安装的候选中文字体:', get_available_chinese_fonts() or '无')
    print('当前生效字体:', setup_chinese_font())
