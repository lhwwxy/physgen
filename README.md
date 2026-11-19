# PhysGen — 物理一致性视频生成流水线

针对视频生成模型物理规则缺失（尺寸失真、"重物轻放"等）的闭环工具：
**规划 → 模拟 → 生成 → 物理审查 → 修复 → 全局逐帧审查/修复**。

详见 `../DESIGN.md`（设计方案）。当前进度：**M1（全 Mock）+ M2（Genesis + Seedance 真实生成）+ M3（LLM 规划 + VLM 审查）+ M4（真实重绘修复）全部完成**，全真实后端端到端验收通过。

真实后端全链路：**LLM 语义规划（glm-5.3-flash）→ Genesis 物理模拟/渲染 → Seedance 图生视频（doubao-seedance-2-5-260628）→ VLM 画面级物理审查（尺寸失真/穿模/重量感/冲击合理性）→ 真实修复（Seedance 带修复约束整段重生成 + Seedream 逐帧图生图重绘）→ 全局逐帧审查修复**。

## 部署与快速开始

### 安装（依赖全部在部署阶段下载，工具本身轻量）

```bash
cd physgen
python3.12 -m venv .venv312 && source .venv312/bin/activate

# 1. 基础安装（仅 ~10MB：Mock 流水线开箱即用，无需任何 API key）
pip install -e ".[dev]"

# 2. 部署体检：检查 Python 版本 / 依赖 / ffmpeg / 密钥 / 配置
physgen setup            # 只体检，给出建议命令
physgen setup --install  # 自动安装缺失的真实后端依赖（Genesis/torch/imageio 等）

# 3. 配置密钥（仓库不含任何明文 key）
cp .env.example .env && vi .env      # 填入你的 PHYSGEN_API_KEY
# 也可以直接用环境变量：export PHYSGEN_API_KEY=sk-xxx
```

### 依赖分层说明

| 层 | 内容 | 何时安装 |
|---|---|---|
| 基础 | pillow/pyyaml/typer/rich/requests | `pip install -e .`（Mock 流水线即可跑） |
| `real` extra | matplotlib / genesis-world / torch / imageio+ffmpeg | `pip install -e ".[real]"` 或 `physgen setup --install`；缺失时各模块自动降级 Mock/2D |

> ffmpeg 无需手动安装：系统有就用系统的，没有则 imageio-ffmpeg 在首次抽帧时自动获取。

### 运行

```bash
# 真实后端全链路（一个 key 复用于规划/生成/审查/修复，模型名全部来自 config.yaml）
physgen run "一块10×10米的土地上盖一间8×6×3米的房子，旁边放一个2×2×1米的石块" --out runs/real
# 每段：LLM 规划 → Genesis 模拟首帧+GT → Seedance 图生视频（720p/4s）
#       → VLM 画面级物理审查 → 真实修复（重生成/重绘）→ 全局逐帧审查修复

# 纯 Mock demo（无需任何 API key）
# 将 config.yaml 中各模块 provider 改为 mock
```

### 配置约定

- **模型名**：`config.yaml` 各段的 `model` 字段是唯一来源（`glm-5.3-flash` / `doubao-seedance-2-5-260628` 等只是当前配置值，换成你自己的即可）；代码不内置任何模型 id，真实 provider 缺配置会直接报错指路
- **密钥**：`config.yaml` 只含 `${PHYSGEN_API_KEY}` 占位引用；实际密钥放在 `.env`（已被 `.gitignore` 排除）。优先级：shell 环境变量 > `.env`。checkpoint 快照（`runs/*/state.json`）已脱敏，不含密钥
- **端点**：各段 `base_url` **必填**（OpenAI 兼容网关或官方 API，如 Ark: `https://ark.cn-beijing.volces.com/api/v3`）；代码不内置任何端点，未配置会直接报错，`physgen setup` 也会体检此项

产出（`runs/<run_id>/`）：
- `report.html` — 审查报告：逐段检查项、异常清单、修复记录、全局逐帧审查结果
- `seg_XX/first_frame.png` — 模拟器渲染的首帧（含 1.7m 人物比例参照）
- `seg_XX/frame_*.png / clip.mp4` — 生成视频（无 ffmpeg 时仅帧序列）
- `final/` — 全局修复后的成片
- `state.json` — checkpoint，支持 `physgen resume runs/<run_id>` 断点续跑

## 架构

五个模块全部面向接口，`config.yaml` 选择实现，API key 走环境变量：

| 模块 | 职责 | Mock（已实现） | 真实后端 |
|---|---|---|---|
| planner | 需求 → SceneSpec（含物理合法性校验/自动修正） | 正则+规则引擎 | **`llm` 已交付**（语义解析；质量/摆放/校验仍由确定性代码兜底） |
| simulator | 首帧渲染 + 控制信号 + 物理真值 GT | 参数化模拟 | **`genesis` 已交付**（真实 3D 物理 + Rasterizer 渲染） |
| generator | 首帧 → 视频段 | 确定性扰动占位视频 | **`seedance` 已交付**（doubao-seedance-2-5-260628，OpenAI Videos 协议，官方 Ark 同适配） |
| auditor | 尺寸/轨迹/落地时长/重量感审查 | 数值真值比对 | **`vlm` 已交付**（抽帧+GT 期望对比，画面级违规检出） |
| repairer | 段级 + 帧级修复 | 标记式修复 | **`seedance` 已交付**（段级 Seedance 重生成 + 帧级 Seedream 图生图重绘；失败自动保留原帧） |

## 测试

```bash
pytest tests/ -v
```

## 里程碑

- [x] M1 全 Mock 端到端流水线 + CLI + checkpoint + HTML 报告
- [ ] M2 Genesis 模拟器（真实 3D 首帧与 GT）
- [ ] M3 LLM 规划器 + VLM 审查器
- [ ] M4 可灵/通用 HTTP 生成与重绘适配器
- [ ] M5 全局审查/修复完善
