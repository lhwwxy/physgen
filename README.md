# PhysGen — 物理一致性视频生成流水线

针对视频生成模型物理规则缺失问题的闭环工具。直接生成的视频常出现**尺寸失真、重物轻放、物体穿模、悬浮失重**等物理错误——PhysGen 用"真实物理模拟 + 画面级审查 + 自动修复"来兜底：

```
用户需求 → LLM 语义规划 → 物理模拟 → 视频生成 → VLM 物理审查 → 自动修复 → 逐帧复审 → 成片
```

## 它能做什么

| 环节 | 实现 | 实测效果 |
|---|---|---|
| **语义规划** | LLM 解析中文需求为结构化场景（物体/尺寸/材质/事件），确定性代码兜底质量计算与合法性校验 | "20×10×12m 办公楼超地块"被自动缩到合法值 |
| **物理模拟** | Genesis 真实 3D 物理引擎：首帧渲染、控制信号、逐帧物理真值（GT） | 首帧含 1.7m 人物比例参照 |
| **视频生成** | Seedance 图生视频（OpenAI Videos 协议，官方 Ark 与兼容网关均可） | 720p / 4s 每段 |
| **物理审查** | VLM 抽帧 + GT 期望值对比，画面级违规检出 | 真实检出：尺寸失真 2 倍、物体嵌墙、动能量化推理（"200kJ 不足以产生爆炸式抛射"） |
| **自动修复** | 段级：带 VLM 修复建议整段重生成；帧级：图像编辑模型逐帧重绘 | 端到端验收：修复后审查全过/仅剩个别项 |

修复预算可配（轮数/帧数封顶），预算耗尽按设计降级接受并在报告中标注。

## 安装

```bash
git clone https://github.com/lhwwxy/physgen.git && cd physgen
python3 -m venv .venv && source .venv/bin/activate

# 基础安装（轻量，Mock 流水线开箱即用，无需任何 API key）
pip install -e .

# 部署体检：检查 Python/依赖/ffmpeg/密钥/配置，可自动安装真实后端依赖
physgen setup            # 只体检，给出建议命令
physgen setup --install  # 自动安装 Genesis/torch/imageio 等（或手动 pip install -e ".[real]"）
```

依赖按需下载，不捆绑：基础层仅 ~10MB；Genesis/torch 等重依赖装在 `real` extra，缺失时各模块自动降级 Mock/2D。ffmpeg 无需手动装（imageio-ffmpeg 首次抽帧时自动获取）。

## 配置

只需两个文件：

**1. `.env`** —— 唯一密钥文件（不入库）：

```bash
cp .env.example .env
# 填入你的 key；所有模块默认共用：
PHYSGEN_API_KEY=sk-xxx
```

**2. `config.yaml`** —— 所有可调参数（provider 选 mock/真实后端、模型名、端点、审查阈值、修复预算、时长分辨率等）：

```yaml
planner:   { provider: llm,      model: <任意 LLM>,           base_url: <OpenAI 兼容端点>, api_key: ${PHYSGEN_API_KEY} }
simulator: { provider: genesis }                                # 本地物理引擎，无需 key
generator: { provider: seedance, model: <视频模型>,           base_url: ..., api_key: ${PHYSGEN_API_KEY} }
auditor:   { provider: vlm,      model: <支持 image_input>,   base_url: ..., api_key: ${PHYSGEN_API_KEY} }
repairer:  { provider: seedance, model: ..., image_model: <图像编辑模型>, base_url: ..., api_key: ${PHYSGEN_API_KEY} }
```

- **最小配置**：一个 key + 各模块一行 `model` + `base_url`；全部改 `mock` 则零配置运行
- **多供应商**：LLM/VLM 与视频生成不是同一家时，各模块 `api_key` 引用不同变量（见 `.env.example`）
- 代码不内置任何模型 id 和端点，缺配置会直接报错指路；`physgen setup` 也会逐项体检

## 使用

```bash
# 全链路运行（真实后端）
physgen run "一块10×10米的土地上盖一间8×6×3米的房子，旁边放一个2×2×1米的石块" --out runs/demo

# 长需求自动分段，支持断点续跑（不重复调用已完成的生成步骤）
physgen resume runs/demo

# 重新渲染审查报告
physgen report runs/demo
```

产出（`runs/<run_id>/`）：

| 文件 | 内容 |
|---|---|
| `report.html` | 审查报告：逐段检查项、异常清单（含物理量化推理）、修复记录、全局逐帧复审 |
| `seg_XX/first_frame.png` | 模拟器渲染的首帧 |
| `seg_XX/clip.mp4` | 生成/修复后的视频段 |
| `final/` | 全局修复后的成片 |
| `state.json` | checkpoint（已脱敏，不含密钥） |

## 架构

五个模块全部面向接口，`config.yaml` 逐模块选择实现，可独立替换：

| 模块 | 职责 | Mock | 真实后端 |
|---|---|---|---|
| planner | 需求 → SceneSpec（校验/自动修正） | 正则+规则 | `llm`（语义解析） |
| simulator | 首帧渲染 + 控制信号 + 物理真值 GT | 参数化模拟 | `genesis`（3D 物理） |
| generator | 首帧 → 视频段 | 占位视频 | `seedance` / `kling` / `http_generic` |
| auditor | 尺寸/轨迹/落地时长/重量感审查 | 数值比对 | `vlm`（画面级） |
| repairer | 段级重生成 + 帧级重绘 | 标记式修复 | `seedance`（重生成）+ 图像编辑（重绘） |

任一真实后端不可用时自动降级 Mock，流水线不中断。

## 作为服务集成到其他平台

流水线核心零改动，附 HTTP 服务化包装（异步任务模式，适合视频制作平台等外部系统对接）：

```bash
pip install -e ".[server]"
PHYSGEN_SERVER_TOKEN=your-secret uvicorn server:app --port 8000   # 或 physgen-server
```

| 接口 | 说明 |
|---|---|
| `POST /tasks` | 提交需求（prompt/config/duration_s），立即返回 task_id |
| `GET /tasks/{id}` | 状态与进度（来自 checkpoint：规划/段进度/全局审查） |
| `POST /tasks/{id}/resume` | 失败或中断的任务断点续跑（不重复调用已完成的生成步骤） |
| `GET /tasks/{id}/result` | 产物清单（成片/各段视频/报告） |
| `GET /tasks/{id}/report` | 审查报告 HTML |
| `GET /tasks/{id}/files?path=...` | 下载任意产物文件（防路径穿越） |

Docker 部署（config.yaml 与 .env 由挂载提供）：

```bash
docker build -t physgen .          # 轻量镜像；--build-arg EXTRAS="server,real" 含 GPU 物理依赖
docker run -d -p 8000:8000 \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  -v $(pwd)/.env:/app/.env:ro \
  -v $(pwd)/runs:/app/runs \
  -e PHYSGEN_SERVER_TOKEN=your-secret physgen
```

> 平台集成提示：单次全链路分钟级耗时与 API 费用，请配合任务队列与预算参数（`budget`）使用；输入内容审核由集成方负责。

## 测试

```bash
pytest tests/ -v   # 30 项：流水线/审查/修复/密钥治理/配置约束
```
