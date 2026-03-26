# model_build

建模根目录：**每个训练框架一个子目录**（当前为 `dr_learner_pipeline/`），共用代码在 **`utils/`**（原 `_model_build/src` 中与流水线相关的模块）。

## 环境

```bash
cd feature_engineering/model_build   # 或 monorepo 中等价路径
pip install -r requirements_dr_pipeline.txt
```

## 运行 DR-Learner 流水线

**工作目录必须是本目录（`model_build` 根）**，以便 `utils` 与 `dr_learner_pipeline` 可被解析。

从 MySQL 拉宽表时须配置连接 JSON（见下节），并指定表名（YAML `db_table` 或 `--db-table`）**或** 使用 YAML `wide_table_path` 读 parquet。

```bash
python -m dr_learner_pipeline.run_pipeline \
  --config dr_learner_pipeline/config/pipeline_grid.yaml \
  --split_dates_path dr_learner_pipeline/example_split_dates.json \
  --db-config-json ../db_config.json \
  --db-table workmagic_bi.your_wide_table
```

`--db-config-json` is optional: if omitted, connection settings follow `LIFT_DB_CONFIG_JSON` or the default `db_config.json` paths below.

其他配置见 `dr_learner_pipeline/config/`（如 `pipeline_fixed.yaml`、`pipeline_scaled_v1_full_feats.yaml`）。

## 数据库配置（JSON，勿提交密钥）

- MySQL 连接仅从 **JSON** 读取，由 `utils/db_config.py` 解析（**不**再依赖 `src2`）。
- **路径优先级**：环境变量 **`LIFT_DB_CONFIG_JSON`**（绝对或相对路径，相对当前工作目录解析）；未设置时默认使用仓库内 **`feature_engineering/db_config.json`**（相对 `utils/db_config.py` 为向上三级目录下的该文件名）。
- JSON 中使用的字段：**`host`**, **`user`**, **`password`**, **`database`**, **`port`**（与现有结构一致）；其余键忽略。`port` 可省略，默认 `3306`。
- **勿将**含真实密码的 `db_config.json` 提交 Git（见仓库 `.gitignore`）。克隆后可复制本目录下的 **`db_config.example.json`** 为 `../../db_config.json`（即 `feature_engineering/db_config.json`）并填写真实值。
- **本地推荐**：在命令行传 **`--db-config-json /path/to/db_config.json`**（在 `main()` 解析参数后、加载 YAML 与连库前即应用，与 Notebook 的 `db_config_json_path=` 等价）。仍可用 **`LIFT_DB_CONFIG_JSON`** 或默认路径，无需重复指定。
- 宽表名 **无代码内默认值**：须在 YAML 写 `db_table` 或命令行传 **`--db-table`**；仅用 parquet 时配置 `wide_table_path` 即可。
- 云上需保证 Notebook / 训练实例与 MySQL 网络互通（VPC / VPN / 白名单）。

## Vertex AI Workbench（概要）

1. 创建 User-managed notebooks 实例（建议多核 CPU）。
2. `git clone <你的仓库>` 后 `cd` 到 **`model_build` 根**（若仓库根即 `model_build`，则直接进入）。
3. `pip install -r requirements_dr_pipeline.txt`
4. 配置 `feature_engineering/db_config.json` 或设置 `LIFT_DB_CONFIG_JSON`（或运行时加 `--db-config-json`）。
5. 执行上文 `python -m dr_learner_pipeline.run_pipeline ...`（含 `--db-table` 或 YAML 中的表名 / `wide_table_path`）。
6. 产物在 `output/<时间戳>/`；可用 `gsutil cp -r output gs://...` 备份。

更细的单元格示例见 **`notebooks/vertex_run.ipynb`**。

## 与 `_model_build` 的关系

当前实现由 **`feature_engineering/_model_build`** 拷贝并改为 **`utils` 包**；`_model_build` 可作备份，稳定后可归档删除。

## 后续

可将 `data_prepare`、`data_merge`、`data_scaling`、`feature_select` 等迁入同一仓库，再逐步整理依赖。
