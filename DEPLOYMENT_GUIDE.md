# 部署指南：本地开发 vs GitHub Pages

## 🎯 问题说明

由于 **本地 Docsify** 和 **GitHub Pages** 的路径解析机制不同，图片路径需要在两种环境之间进行转换：

- **本地 Docsify**: `../assets/image.png` (相对路径)
- **GitHub Pages**: `docs/assets/image.png` (从项目根目录的路径)

## 🔄 环境切换工具

### 1. 为 GitHub Pages 部署准备

```bash
# 转换为 GitHub Pages 路径格式
python fix_github_pages_paths.py
```

**转换效果**：
- `../assets/image.png` → `docs/assets/image.png`
- `../results/image.png` → `results/image.png`

### 2. 恢复本地开发环境

```bash
# 转换回本地 Docsify 路径格式
python fix_local_docsify_paths.py
```

**转换效果**：
- `docs/assets/image.png` → `../assets/image.png`
- `results/image.png` → `../results/image.png`

## 📋 完整部署流程

### 🚀 部署到 GitHub Pages

1. **准备部署**：
   ```bash
   python fix_github_pages_paths.py
   ```

2. **提交到 GitHub**：
   ```bash
   git add .
   git commit -m "Fix image paths for GitHub Pages"
   git push origin main
   ```

3. **启用 GitHub Pages**：
   - 进入 GitHub 仓库设置
   - Pages → Source → Deploy from a branch
   - 选择 `main` 分支和 `/docs` 文件夹

### 💻 恢复本地开发

部署完成后，如果需要继续本地开发：

```bash
# 恢复本地路径
python fix_local_docsify_paths.py

# 启动本地服务器
docsify serve docs
```

## 🎨 HTML 图片标签优势

使用 HTML `<img>` 标签而不是 Markdown `![]()`：

```html
<img src="docs/assets/image.png" 
     alt="图片描述" 
     style="max-width: 100%; 
            height: auto; 
            display: block; 
            margin: 20px auto; 
            border: 1px solid #eee; 
            border-radius: 4px; 
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
```

**优势**：
- ✅ **避免 LaTeX 插件冲突**
- ✅ **更好的样式控制**
- ✅ **响应式设计**
- ✅ **统一的视觉效果**

## 🔧 工具脚本说明

### `fix_github_pages_paths.py`
- **用途**: 为 GitHub Pages 部署准备路径
- **转换**: 相对路径 → 绝对路径
- **使用时机**: 部署前

### `fix_local_docsify_paths.py`
- **用途**: 恢复本地开发环境
- **转换**: 绝对路径 → 相对路径
- **使用时机**: 部署后继续本地开发

## 📁 路径对照表

| 文件位置 | 本地 Docsify | GitHub Pages |
|---------|-------------|-------------|
| `docs/visualizations/` | `../assets/image.png` | `docs/assets/image.png` |
| `docs/experiments/` | `../assets/image.png` | `docs/assets/image.png` |
| `docs/` 根目录 | `assets/image.png` | `docs/assets/image.png` |
| 项目根目录 | `docs/assets/image.png` | `docs/assets/image.png` |
| Results 图片 | `../results/image.png` | `results/image.png` |

## ⚠️ 注意事项

1. **部署前必须运行**: `python fix_github_pages_paths.py`
2. **本地开发前运行**: `python fix_local_docsify_paths.py`
3. **不要手动修改路径**: 使用脚本确保一致性
4. **提交前检查**: 确保路径格式正确

## 🎯 当前状态

**当前路径格式**: GitHub Pages (准备部署)
- 图片路径: `docs/assets/image.png`
- 可以直接提交到 GitHub

如需本地开发，请运行：
```bash
python fix_local_docsify_paths.py
``` 