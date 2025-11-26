./generated里有两个比较大的文件不好上传：
1. hnsw.index是hnsw索引文件，超过了1G，我在想办法
2. lookup.npy，用来通过token_id反向查找token，需要手动执行build-lookup.py来构建

那个graph.graphml（即上下文共现图）只是测试版，我之后会弄一个更好的，但和这个在调用方式上没有区别

有些库可能要先安装：
```
pip install faiss-cpu
pip install igraph
pip install json
pip install numpy
```
