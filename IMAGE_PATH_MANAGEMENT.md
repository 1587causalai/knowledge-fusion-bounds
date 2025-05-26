# 图片路径管理指南

本项目使用 Docsify 进行文档展示，支持本地开发和 GitHub Pages 部署。由于两种环境对图片路径的要求不同，我们提供了自动化脚本来管理路径切换。

## 问题说明

- **本地开发**: Docsify 从 `docs/` 目录提供服务，图片路径需要使用相对路径（如 `../assets/image.png`）
- **GitHub Pages**: 从项目根目录提供服务，图片路径需要使用绝对路径（如 `docs/assets/image.png`）

## 解决方案

### 1. 简化的 HTML 标签

我们已经将复杂的 HTML `<img>` 标签简化为：
```html
<img src="path/to/image.png" alt="图片描述">
```

### 2. 自动化路径管理

使用统一的管理脚本来切换路径：

#### 本地开发模式
```bash
python manage_image_paths.py local
```
然后运行：
```bash
docsify serve docs
```

#### GitHub Pages 部署模式
```bash
python manage_image_paths.py github
```
然后提交并推送到 GitHub。

## 工作流程

### 本地开发
1. 切换到本地模式：`python manage_image_paths.py local`
2. 启动本地服务：`docsify serve docs`
3. 在浏览器中访问 `http://localhost:3000` 查看效果

### 部署到 GitHub Pages
1. 切换到 GitHub 模式：`python manage_image_paths.py github`
2. 提交更改：`git add . && git commit -m "Update for GitHub Pages"`
3. 推送到 GitHub：`git push`

## 注意事项

- 在切换模式之前，请确保当前的更改已经保存
- 建议在部署前先在本地测试
- 如果添加新的图片，请使用简化的 HTML 标签格式
- 图片文件应放在 `docs/assets/` 目录下

## 脚本说明

- `simplify_img_tags.py`: 简化复杂的 HTML img 标签
- `fix_local_docsify_paths.py`: 修复本地 Docsify 的图片路径
- `fix_github_pages_paths.py`: 修复 GitHub Pages 的图片路径
- `manage_image_paths.py`: 统一的路径管理脚本（推荐使用） 