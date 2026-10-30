DexYCB 数据集中管理在本目录。

- `sequences/<object>/<sequence>/mano.npz`：现有两批共十条打包原始序列。index.json 的 path 均相对于本目录。
- `raw/official`：已有原始样本、标定和物体资产。
- `prepared/sugar_box`：默认可运行输入，含手/物体参考、接触指导、模型和物理参数 model.yaml。
- `additional_source_manifest.json`、`additional_validation.json`：原第二批样本的来源与校验记录。

原始 MANO 打包序列不是完整运行输入。统一入口只接受已准备的模型和参考，不会隐式生成或补造缺失的转换结果。
