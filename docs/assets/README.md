# 资源文件目录

本目录包含项目的所有静态资源文件，包括图片、HTML文件和生成这些资源的代码。

## 目录结构

### 图片文件 (*.png)
- `position_only_*.png` - 位置参数扰动相关的可视化图片
- `scale_only_*.png` - 尺度参数扰动相关的可视化图片  
- `both_params_*.png` - 双参数扰动相关的可视化图片
- `multivariate_*.png` - 多维情况相关的可视化图片
- `extreme_cases_*.png` - 极端情况相关的可视化图片
- `optimal_constant_*.png` - 最优常数探索相关的可视化图片

### HTML文件 (*.html)
- `counterexamples.html` - 反例分析的交互式页面
- `inequality_holds_summary.html` - 不等式成立情况的汇总页面
- `optimal_constant_by_N.html` - 按N值分组的最优常数分析

### 代码文件
- `generate_visualizations.py` - 生成所有可视化图片和HTML文件的Python脚本

## 使用说明

### 生成可视化资源

要重新生成所有的可视化资源，请运行：

```bash
cd docs/assets
python generate_visualizations.py
```

### 在文档中引用图片

在 `docs/visualizations/` 目录下的Markdown文档中，使用相对路径引用图片：

```markdown
<img src="../assets/图片文件名.png" alt="图片描述">
```

## 文件职责分离

- **docs/assets/** - 存放所有静态资源文件（图片、HTML、生成代码）
- **docs/visualizations/** - 存放可视化相关的Markdown文档
- **docs/theory/** - 存放理论分析相关的Markdown文档
- **docs/experiments/** - 存放实验研究相关的Markdown文档

这种结构确保了内容与资源的清晰分离，便于维护和管理。 