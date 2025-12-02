# Markdown查看器使用说明

## 功能

提供了一个Web界面来查看Markdown格式的分析报告，支持：
- 在浏览器中查看Markdown文件
- 自动渲染Markdown为HTML
- 支持文件选择和文件列表
- 美观的样式和布局

## 使用方法

### 方法一：使用HTTP服务器（推荐）

1. **启动服务器**：

```bash
cd /home/llminference/syq/dyna-spec/analysis
python view_server.py
```

2. **在浏览器中打开**：

访问 `http://localhost:8000`

3. **自定义端口**：

```bash
python view_server.py --port 8080
```

4. **指定文件目录**：

```bash
python view_server.py --directory /path/to/markdown/files
```

### 方法二：直接打开HTML文件

1. **打开HTML文件**：

直接在浏览器中打开 `view_markdown.html` 文件

2. **选择Markdown文件**：

点击"选择Markdown文件"按钮，选择要查看的 `.md` 文件

## 功能特性

- ✅ 自动渲染Markdown为HTML
- ✅ 支持表格、代码块、列表等Markdown语法
- ✅ 响应式设计，适配不同屏幕尺寸
- ✅ 美观的样式和配色
- ✅ 支持文件列表快速选择
- ✅ 支持从服务器加载文件

## 界面说明

- **文件选择器**：可以选择本地Markdown文件
- **文件列表**：显示当前目录下的所有 `.md` 文件（服务器模式）
- **内容区域**：显示渲染后的Markdown内容

## 注意事项

- 如果使用HTTP服务器模式，确保Markdown文件在服务器目录下
- 直接打开HTML文件时，只能选择本地文件
- 服务器模式支持通过URL参数指定文件：`http://localhost:8000?file=result.md`

## 示例

```bash
# 启动服务器
cd /home/llminference/syq/dyna-spec/analysis
python view_server.py

# 在浏览器中访问
# http://localhost:8000

# 查看特定文件
# http://localhost:8000?file=result_num_3000.md
```

