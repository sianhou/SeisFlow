---
name: research-direction
description: 创建或更新 SeisFlow 的独立研究方向文档时使用（文件以 directions_ 开头，不包括 directions_board.md），详细记录方向判断、实验、Git 版本、代码脚本、配置、结果和后续条件。
---

# 研究方向文档规范

## 用途与命名

每个研究方向使用一个独立文档，文件名为 `research_logs/directions_<name>.md`，例如 `directions_shared_noise.md`。

`<name>` 使用简短、稳定的英文名称。同一方向不要因为结论或级别变化而更换文件名。

方向分为三级：

- **重要**：当前主线、关键结论，或者明显改善效果的方向。
- **保留**：有一定价值，但证据不足、效果一般或当前优先级较低。
- **暂停**：效果不好或目前不值得继续。保留已有记录，并写明重新考虑的条件。

## 格式

```markdown
# 研究方向：方向名称

- 级别：重要 / 保留 / 暂停
- 最后更新：YYYY-MM-DD

## 研究目的

- 要解决的问题或验证的假设。

## 当前结论

- 当前判断：
- 主要证据：
- 局限或尚未确认的问题：

## 实验记录

### YYYY-MM-DD：实验名称

- 相关月志：[YYYY-MM-DD](YYYYMM.md#yyyy-mm-dd)
- 目的：
- 方法与配置：
- 对照：
- 结果：
- 观察：
- 解释：

#### 版本与实现

- Git：`<commit>`（clean / dirty）
- 未提交相关文件：无 / `path/to/file`
- 入口脚本：`path/to/script.sh`
- 主要代码：`path/to/code.py`
- 关键配置：模型、checkpoint、EMA、数据集、随机种子等
- 结果目录：`path/to/output`

## 下一步

1. 
```

## Git 与复现信息

新增正式实验记录前执行：

```bash
git rev-parse --short HEAD
git status --short
```

- 工作区干净时记录 `<commit> (clean)`。
- 有未提交修改时记录 `<commit> (dirty)`，并列出与该实验有关的修改文件。不要把未提交代码描述成该 commit 已包含的内容。
- 代码和脚本使用仓库相对路径，不能只写文件名或口头描述。
- 正式实验至少记录入口脚本、主要实现、checkpoint、数据集、关键参数和结果目录。
- 不同实验分别保留当时的 Git 和实现信息，不能用新版本覆盖旧实验记录。

## 更新要求

- 新实验追加到“实验记录”，不要只改当前结论而丢失过程。
- 更新“当前结论”时，以已有实验记录为依据，并同时更新日期。
- 指标尽量写出对照值、实验值和变化量。
- 明确区分“观察”和“解释”，推测不能写成已证明结论。
- 方向级别变化时，在对应研究月志中简要记录原因并链接回本文档。
- 新建方向，或方向的名称、级别、当前判断、下一步和更新时间发生变化时，同步更新 `research_logs/directions_board.md`。更新前阅读 `research_logs/skills/research-direction-board/SKILL.md`。
- 效果不好的方向不能删除，标记为“暂停”并写明重新考虑的条件。
- 图片放入 `research_logs/images/YYYYMMDD/`，并使用相对路径引用。
