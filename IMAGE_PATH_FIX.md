# 图片路径修复说明

## 问题描述

在 Markdown 文档中，图片路径的引用方式会影响在不同环境下的显示效果：

- **本地环境**：相对路径基于当前文件所在目录
- **GitHub Pages**：路径解析基于项目根目录

这导致了在本地能正常显示的图片，在 GitHub Pages 上无法显示的问题。

## 解决方案

采用**统一的绝对路径**（从项目根目录开始），确保在本地和 GitHub Pages 上都能正常显示。

### 标准路径格式

```markdown
# ✅ 正确格式（从项目根目录开始，无前导斜杠）
![图片描述](docs/assets/image.png)
![图片描述](results/image.png)

# ❌ 错误格式
![图片描述](../assets/image.png)          # 相对路径
![图片描述](/docs/assets/image.png)       # 带前导斜杠
![图片描述](./assets/image.png)           # 当前目录相对路径
```

## 自动修复工具

项目中包含了 `fix_image_paths.py` 脚本，可以自动修复所有 Markdown 文件中的图片路径。

### 使用方法

```bash
# 在项目根目录运行
python fix_image_paths.py
```

### 脚本功能

1. **扫描所有 Markdown 文件**：包括根目录和所有子目录
2. **识别图片引用**：支持 `.png`, `.jpg`, `.jpeg`, `.gif`, `.svg` 格式
3. **路径转换**：
   - `../assets/image.png` → `docs/assets/image.png`
   - 相对路径 → 从项目根目录的绝对路径
   - 移除前导斜杠（GitHub Pages 兼容性）
4. **安全处理**：只修改需要修复的文件，保留原有正确路径

## 修复结果

运行脚本后，以下文件已被修复：

- `docs/assets/README.md`
- `docs/experiments/experiment_results.md`
- `docs/visualizations/both_parameters.md`
- `docs/visualizations/extreme_cases.md`
- `docs/visualizations/multivariate.md`
- `docs/visualizations/optimal_constant.md`
- `docs/visualizations/position_parameter.md`
- `docs/visualizations/scale_parameter.md`

## 验证方法

### 本地验证

1. **使用 Markdown 预览器**：在 VS Code 或其他编辑器中预览
2. **本地服务器**：使用 `python -m http.server` 启动本地服务器测试

### GitHub Pages 验证

1. 提交修改到 GitHub
2. 检查 GitHub Pages 部署的网站
3. 确认所有图片都能正常显示

## 最佳实践

1. **新增图片时**：直接使用从项目根目录开始的路径
2. **定期检查**：可以定期运行修复脚本确保路径一致性
3. **文档规范**：在团队中统一使用绝对路径格式

## 技术细节

### 路径解析规则

- **docs/ 目录下的文件**：`../assets/` → `docs/assets/`
- **根目录文件**：保持现有的 `docs/assets/` 和 `results/` 路径
- **其他相对路径**：自动解析为从项目根目录的绝对路径

### GitHub Pages 兼容性

- 不使用前导斜杠（避免 `/docs/assets/` 格式）
- 确保路径从项目根目录开始
- 支持 Jekyll 和 Docsify 等静态站点生成器

## 故障排除

如果图片仍然无法显示：

1. **检查文件是否存在**：确认图片文件在指定路径下存在
2. **检查文件名大小写**：确保文件名大小写完全匹配
3. **检查特殊字符**：避免文件名中包含空格或特殊字符
4. **重新运行脚本**：`python fix_image_paths.py`

## 相关文件

- `fix_image_paths.py`：自动修复脚本
- `IMAGE_PATH_FIX.md`：本说明文档
- 所有 `.md` 文件：包含图片引用的 Markdown 文档 