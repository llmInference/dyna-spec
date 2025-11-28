./generated里有两个比较大的文件不好上传：
1. hnsw.index是hnsw索引文件，超过了1G，我放到了百度网盘 https://pan.baidu.com/s/14vCafsfn1hIrBZOalAqlcw?pwd=8085
2. lookup.npy，用来通过token_id反向查找token，需要手动执行build-lookup.py来构建

有些库可能要先安装：
```
pip install faiss-cpu
pip install igraph
pip install json
pip install numpy
```
