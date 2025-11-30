# 推测解码输出分析工具

## 功能

这个工具集用于分析推测解码（Speculative Decoding）的输出结果，包括：

1. **分析脚本** (`analyze_spec_output.py`)：
   - 整理草稿模型每轮次的输出：显示每轮草稿模型生成的所有token，并标记哪些被接受、哪些被拒绝
   - 整理目标模型的输出：显示每轮目标模型实际选择的token，并与草稿模型进行对比
   - 生成Markdown格式的分析报告

2. **Web查看器** (`view_server.py` + `view_markdown.html`)：
   - 在浏览器中查看Markdown分析报告
   - 美观的界面和自动渲染
   - 支持文件选择和快速浏览

## 使用方法

### 方法一：从标准输入读取（仅打印到控制台）

```bash
cat output.txt | python analyze_spec_output.py
```

或者

```bash
python analyze_spec_output.py < output.txt
```

### 方法二：从文件读取（仅打印到控制台）

```bash
python analyze_spec_output.py output.txt
```

### 方法三：保存结果到Markdown文件

```bash
# 从文件读取并保存到markdown
python analyze_spec_output.py output.txt -o result.md

# 从标准输入读取并保存到markdown
cat output.txt | python analyze_spec_output.py -o result.md
```

**注意**: 使用 `-o` 或 `--output` 参数指定输出文件路径，结果会保存为格式化的Markdown文件。

## 输出说明

### 控制台输出

脚本会在控制台输出简化的分析结果，包括：
- 基本信息（最终生成文本、平均接受长度、总轮数）
- 如果指定了输出文件，会显示保存路径

### Markdown文件输出

当使用 `-o` 参数时，会生成详细的Markdown格式报告，包含：

1. **基本信息**
   - 最终生成文本
   - 平均接受长度
   - 总轮数
   - 生成时间

2. **草稿模型输出（按轮次）**
   - 每轮草稿模型生成的所有token
   - 使用**粗体**标记被接受的token，使用~~删除线~~标记被拒绝的token
   - 拒绝详情表格，显示草稿token和目标token的对比（包括Token ID）

3. **目标模型输出（按轮次）**
   - 每轮目标模型实际选择的token序列
   - 接受的token（草稿模型和目标模型都选择了）
   - 拒绝的token对比列表

4. **对比总结表格**
   - 草稿模型总token数
   - 接受的token数
   - 目标模型总token数
   - 接受率

## 输入格式

脚本期望输入为JSON格式，每行一个JSON对象，包含以下字段：

- `text`: 最终生成的文本
- `draft_avg_accept_length`: 平均接受长度
- `spec_total_rounds`: 总轮数
- `spec_round_details`: 每轮详情数组，每个元素包含：
  - `draft_tokens`: 草稿模型生成的token列表
  - `accept_length`: 接受的token数量
  - `rejections`: 拒绝信息列表

## 示例

假设有一个输出文件 `output.json`，内容如下：

```json
{"text":"5,6,7,8,9,","draft_avg_accept_length":0.0,"spec_total_rounds":9,"spec_round_details":[...]}
```

### 示例1：仅查看控制台输出

```bash
python analyze_spec_output.py output.json
```

### 示例2：保存到Markdown文件

```bash
python analyze_spec_output.py output.json -o analysis_result.md
```

生成的 `analysis_result.md` 文件包含完整的分析报告，可以使用任何Markdown阅读器查看。

### 示例3：处理多个JSON对象

如果输入文件包含多行JSON（每行一个JSON对象），脚本会：
- 如果指定了 `-o` 参数且只有一行数据：保存到指定文件
- 如果指定了 `-o` 参数且有多行数据：合并所有结果到一个文件
- 如果没有指定 `-o` 参数：为每行数据生成单独的文件（`output_1.md`, `output_2.md`等）

## 在网页中查看Markdown报告

生成Markdown文件后，可以使用Web查看器在浏览器中查看：

### 方法一：使用HTTP服务器（推荐）

```bash
# 启动服务器
cd /home/llminference/syq/dyna-spec/analysis
python view_server.py

# 或使用快速启动脚本
./start_viewer.sh
```

然后在浏览器中访问：`http://localhost:8000`

### 方法二：直接打开HTML文件

直接在浏览器中打开 `view_markdown.html` 文件，然后选择要查看的Markdown文件。

### 查看器功能

- ✅ 自动渲染Markdown为HTML
- ✅ 支持表格、代码块、列表等
- ✅ 响应式设计
- ✅ 文件列表快速选择
- ✅ 美观的样式和配色

详细说明请参考 [README_VIEWER.md](README_VIEWER.md)

