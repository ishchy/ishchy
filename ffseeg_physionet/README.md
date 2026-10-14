# PhysioNet EEGMMIDB：独立受试者实验

## 研究定位

本实验是**新数据集上重新训练模型后的受试者独立验证**，不是 UCI 模型直接迁移到 PhysioNet 的零样本跨数据集测试。PhysioNet EEGMMIDB 仅有 R01 睁眼与 R02 闭眼的单独基线记录，**眼状态与录制运行/次序完全共线**。因此只能把本实验解释为不同实验运行中的睁/闭眼静息条件识别，不能单独证明真正的实时眼状态识别或因果神经生理机制。

官方数据：https://physionet.org/content/eegmmidb/1.0.0/ 。109个受试者、64通道、160 Hz；每人 R01 和 R02 各约一分钟。只下载这两个运行，不需要下载全套14个运行。

## 方案

- 从每位受试者的 R01 和 R02 EDF 读取14个电极（与原稿布局对应）。使用无重叠2秒窗口计算 log[8-13 Hz 功率 / 4-30 Hz 功率]，节点输入仍为1维。
- 10折**按受试者划分**，所有受试者的训练/验证/测试绝不交叉。训练标准化仅使用训练受试者。
- 方法：全节点 FullGraphConv、训练集互信息固定选择8节点的后编码 MI-PostGCN、输入端 MI-InputGCN、原FFSEEG-GNN、随机固定后编码 Random-PostGCN。
- 前编码子图在8节点上重新归一化；后编码14个电极都经过两层GCN后仅选8个后端表征；每个被试拥有成对睁/闭眼窗口；输入信息量本来不同，不能把差异归结为选择操作本身。
- 每折固定模型超参；最多100轮，软门控与硬微调各自最多100轮，早停依赖训练受试者之外的验证受试者（不能使用测试受试者）。
- 输出：逐折表现、逐受试者OOF预测、受试者级统计、配对bootstrap置信区间、每折Top8节点、数据来源和运行元数据、SHA256SUMS。

## 执行：建议 GitHub Actions

将下列3个 Python 文件以及 `.github/workflows/physionet_subject_cv.yml` 放到本仓库的 ffseeg_physionet/ 目录（工作流放到 .github/workflows/）并提交：

- `ffseeg_repro_experiments.py`
- `ffseeg_physionet_external.py`
- `physionet_extended_cv.py`

进入 GitHub 仓库 **Actions → PhysioNet EEGMMIDB subject held-out comparison → Run workflow**，保留 `subjects_end=109`, `folds=10`, `epochs=100`，等待完成后下载 Artifacts 并将 ZIP 上传对话，进行真实结果审查及论文合并。

外网受限时，也可以用浏览器将 `https://physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf` 等文件保存为 `cache/eegmmidb/S001/S001R01.edf`；脚本会使用本地缓存，不重复下载。

本地执行：

```bash
python -m pip install numpy scipy pandas scikit-learn torch mne edfio
python ffseeg_physionet/physionet_extended_cv.py --out results --n-splits 10 --max-epochs 100 --cache-dir cache/eegmmidb
```

## 自检说明

`python ffseeg_physionet/physionet_extended_cv.py --smoke --n-splits 2 --max-epochs 2 --out smoke_results` 使用合成数据，仅验证程序逻辑，**绝对不可写入论文作为真实 EEG 性能**。只有真实实验文件中的 `RUN_COMPLETE.txt` 内容为 `Real EEG result` 时，方可进入人工审核，且需要核实EDF下载、窗口统计和受试者分组。

## 为什么现在没有真实跨被试结果？

当前 ChatGPT 计算环境无法解析 `physionet.org` 域名，EDF 文件未下载。以前的UCI EEG Eye State压缩包不包含本数据集。未获取PhysioNet EDF之前，不会报告任何新的真实交叉验证精度数值。