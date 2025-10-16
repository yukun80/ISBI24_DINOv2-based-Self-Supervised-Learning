# 模块名：AM²P（Area-balanced Multi-scale Prototype Pooling）

## 设计哲学（第一性原理）

1. **代表性 > 置信阈**：原型的本质是近似类别条件分布的若干“模态中心”。与其用高阈值（0.95）追求“干净像素”，不如用**面积均衡采样**确保每个空间模态（特别是小而散的区域）都被代表。
2. **尺度覆盖 > 固定核**：遥感目标天然多尺度，固定核的平均会产生偏差。必须**多尺度地**构造局部统计，哪怕每个尺度只是一种**低成本**的方框平均。
3. **少即是多**：在支持端一次性完成区域统计与尺度汇聚；在查询端只保留**余弦匹配 + 线性融合**两步，不引入边界分支、门控或额外的高阶变换。

---

## 模块输入/输出

* 输入：支持特征 `F_s ∈ ℝ^{Hs×Ws×C}`，支持二值掩码 `M_s ∈ {0,1}^{Hs×Ws}`；查询特征 `F_q ∈ ℝ^{Hq×Wq×C}`（来自同一骨干的多尺度或单尺度）。
* 输出：查询前景得分图 `S_q ∈ ℝ^{Hq×Wq}`（或两类 logits）。

---

## 核心流程（替换 ALP 的“AvgPool2d+0.95”）

### 1) 面积均衡锚点（Area-balanced Anchors）

* 在 `M_s` 的**每个连通域**上计算面积 `A_k`。
* 为该域分配锚点数：
  `n_k = clamp(⌈α·log(1 + A_k)⌉, 1, N_max)`（建议 α=1.2，N_max=8）。
* 在该域内做**均匀格点或 farthest-point**采样得到锚点集合 `{a_m}`，保证**小目标至少 1 个锚点**、大目标不会“霸屏”。

> 只基于掩码做 O(HW) 连通域统计 + 轻量采样，GPU/CPU 均快。

### 2) 多尺度局部原型（Multi-scale Local Means）

* 预设少量尺度半径集 `R = {r₁, r₂, r₃}`（如 `{4, 8, 16}` 个像素，按分辨率线性缩放）。
* 对每个锚点 `a_m`、每个尺度 `r ∈ R`，在支持特征上用**方框平均**（积分图或深度可分离卷积实现）计算：
  `p_{m,r} = mean_{x ∈ N(a_m, r) ∩ M_s=1} F_s(x)`（若前景像素数不足 `θ_min` 则跳过）。
* **尺度自选（无门控）**：为同一锚点保留最多一个尺度的原型：
  选择满足前景占比最高且局部特征方差最小的尺度
  `r* = argmax_r  ρ_{m,r} / (ε + Var_{m,r})`，得到 `p_m = p_{m,r*}`，并记录权重 `w_m = ρ_{m,r*}`（ρ 为窗口内前景占比）。

> 这一步只做**均值与方差**的局部统计；无需边界估计与学习门控。

### 3) 全局/小样补偿（Global & Minority Safeguard）

* **全局原型**：`p_g = mean_{M_s=1} F_s(x)`；对**极小连通域**（面积 < τ_area）的锚点，若步骤 2 被跳过，则直接赋 `p_m ← p_g`（确保小目标不被丢）。
* **去冗余**：对 `{p_m}` 做一次**k-center on sphere**（余弦距离）压到 `M_max`（如 48），抽样优先保留**不同连通域**与**不同尺度来源**的原型（通过分桶轮询实现）。得到最终原型集 `P = { (p_i, w_i) }_{i=1..M}` 和 `p_g`。

### 4) 查询端匹配与融合（Linear, No Frills）

* 计算查询特征与原型的**余弦相似**：
  `s_i(x) = cos(F_q(x), p_i)`；`s_g(x) = cos(F_q(x), p_g)`
* **线性加权融合**（不做复杂门控）：
  `S_q(x) = softmax_τ(  β · s_g(x)  +  Σ_i  ŵ_i · s_i(x) )`
  其中 `ŵ_i = w_i / Σ w_i`，`β∈[0,1]` 控制全局原型比重（缺省 0.3），`τ` 为温度（缺省 0.07）。
* 输出二分类 logits 或概率。

---

## 复杂度评估

* 支持端：连通域 + 采样 O(HW)，3 个尺度的**方框均值/方差**可用积分图近似 O(HW·|R|)，常数极小。
* 查询端：与 ALP 同阶的**M 个原型相似度**（点积 + 归一），不含任何边界分支、门控 MLP 或多头注意力。
* 典型设置（`|R|=3, M_max=48`）下，总 FLOPs 与显存开销 ≈ 原 ALPNet 的 0.8~1.0×。

---

## 与原 ALPNet 的关键差异

* **去掉**：AvgPool2d 固定核 + FG_THRESH=0.95；不做边界/门控。
* **保留并强化**：局部原型思想，但用**面积均衡 + 多尺度自选**替代“固定核 + 高阈值”。
* **新增**：k-center 降冗余（常数级），全局/小样补偿（免小目标缺席）。

---

## 与项目参考思路的“轻借鉴”

* 借鉴 **像素级匹配/对齐**（如像素相关图的朴素范式）：我们用**余弦相似**直连，不引入重注意力或变换头。
* 借鉴 **双重匹配/关系空间** 的朴素思想：通过“局部（多模态）+ 全局”的**双轨原型**完成“特征空间的双重覆盖”，仍保持线性融合。

> 这些借鉴只保留“可解释的简洁核”，避免复杂的域变换或代价昂贵的对齐网络。

---

## 接口与落地（可直接替换 ALP）

```python
class AM2P(nn.Module):
    def __init__(self, radii=(4,8,16), alpha=1.2, nmax_comp=8, mmax=48,
                 theta_min=8, beta=0.3, tau=0.07):
        super().__init__()
        self.radii = radii
        self.alpha = alpha
        self.nmax_comp = nmax_comp
        self.mmax = mmax
        self.theta_min = theta_min
        self.beta = beta
        self.tau = tau

    def build_prototypes(self, F_s, M_s):
        # 1) 连通域 & 面积均衡锚点
        # 2) 多尺度方框均值/方差 -> 尺度自选 -> (p_m, w_m)
        # 3) 全局/小样补偿 + k-center -> P, p_g
        # return P (M×C, weights M), p_g (C,)
        ...

    def forward(self, F_s, M_s, F_q):
        P, W, p_g = self.build_prototypes(F_s, M_s)
        # 余弦相似 + 线性融合 + 温度
        # S_q = softmax_tau(beta*cos(q,p_g) + sum_i w_i*cos(q,p_i))
        ...
        return logits
```

* **替换位置**：与 ALP 的 `build_local_prototypes(...)`/`forward(...)` 完全同构，可“即插即用”。
* **实现提示**：积分图 2D 方框均值/方差、连通域用 `scipy.ndimage` 或自写并行版；k-center 用贪心 farthest-point 即可。

---

## 推荐超参（起步）

* `radii=(4,8,16)`；`alpha=1.2`；`nmax_comp=8`；`mmax=48`；`theta_min=8`；`beta=0.3`；`tau=0.07`。
* 若前景极稀疏：提高 `beta` 到 0.5，以增强全局原型兜底。

---

## 最小消融表（建议）

1. 固定核 + 高阈值（原 ALP）
2. **+ 面积均衡锚点**
3. **+ 多尺度自选**
4. **+ k-center 去冗余**
5. **+ 全局/小样补偿**
   记录 mIoU/Dice、小目标召回（<0.5% 图幅）、推理耗时/显存。

---

### 一句话总结

**AM²P** 用“面积均衡 + 多尺度自选 + 线性融合”三把螺丝刀，把原 ALP 的高阈值与固定核的刚性，替换成对遥感更友好的**低复杂度**统计原型构建方式；没有边界分支、没有重门控，却在**小目标、形态多样**和**分布不均**上显著更稳。你把它塞到现有 few-shot 二分类框架里，就是一枚干净利落的替代件。


# 补充说明：

太好了，我们把“AM²P（Area-balanced Multi-scale Prototype Pooling）”落地到你当前框架（现已由 `models/am2p.py` 提供）上，并逐条把你关心的实现细节钉牢。思路延续 ALPNet 的“原型匹配”范式，但把“固定池化 + 高阈值”换成**面积均衡 + 多尺度局部统计**，并保证与 DINO/ALP 风格的两通道 logits 接口完全兼容（你项目就是在 ALPNet 思想上嫁接 DINOv2 的，保持即插即用会更稳 ([arXiv][1])）。

---

## 1) 连通域锚点与对数面积配额（CC + log-area）

**结论**：思路正确，补齐实现细节与可退化路径。

* **CC 例程选择**

  * **首选 GPU 纯 PyTorch**：二维二值掩码的 **Union-Find（并查集）**（每像素 4-邻接），实现上：按行并列扫描，把 `(y,x)` 与 `(y,x-1)`、`(y-1,x)` 逐步合并；最后 `label = find(idx)` reshape 回 H×W。对 512×512 的 mask，耗时可控、零拷贝。
* **锚点抽样**

  * **默认：网格均匀采样（grid-based）**：对每个连通域的外接框以 `stride = max(⌊√(Area_k / n_k)⌋, s_min)` 生成候选格点，只保留落在该域掩码内的点，直到凑够 `n_k`；小域通常只需 1 个点。
  * **可选：FPS（farthest-point）**：当域形状极不规则时更稳。先用网格产生 2–3 倍候选，再在候选上做欧氏距离 FPS 下采样到 `n_k`。
* **配额**：`n_k = clamp(ceil(α·log(1 + Area_k)), 1, N_max_comp)`（默认 `α=1.2, N_max_comp=8`），保证**小域有代表**、大域不过度主导。
* **多 shot 合并时机**：先**在每张支持图内**做 CC 与锚点采样，再把所有原型合并（见第 5 节），避免把本应分开的实例在 CC 前就“粘”在一起。

---

## 2) 多尺度池化与高效 box filter（半径集 R）

**结论**：思路正确，给出两种 GPU 方案与“低阳性”窗口的处理。

* **尺度与窗口**：半径集 `R={4,8,16}`（像素），用 **方框均值** 与 **方框方差**（均在 **支持特征** 上算；掩码只用于筛选前景）。
* **高效实现**（二选一）

  1. **F.unfold**：对每个 `r` 做一次 `unfold(k=(2r+1))` 拆出局部块，随后 `(x*mask).sum()/mask.sum()` 计算均值；Var 用二阶矩同理。优点：全 PyTorch；缺点：一次性展开占显存较大。
  2. **积分图（integral image）**：用 `cumsum` 做二维积分图，`box_sum = I(x2,y2)-I(x1,y2)-I(x2,y1)+I(x1,y1)`，对每个通道 O(1) 取均值/方差；对多尺度非常省时省显存（推荐）。
* **窗口阳性像素阈**：若 `|{x∈N(a_m,r): M_s(x)=1}| < θ_min`（默认 8 像素），**跳过该尺度**；若所有尺度都不达标，**回退到全局原型**（见下节）。
* **尺度自选**：同一锚点多尺度候选里，选择
  `r* = argmax_r  ρ_{m,r}/(ε + Var_{m,r})`，其中 `ρ` 是窗口的阳性占比，`Var` 是通道方差（或 Frobenius 范数），得到局部原型 `p_m` 与权重 `w_m=ρ_{m,r*}`。

---

## 3) K-center 截断与权重归一（去冗余与可复现）

**结论**：思路正确，补充确定性细节与“跨组件”的公平性。

* **K-center/Farthest-First**（球面余弦距离）：

  * **确定性 seed**：取**权重最大的原型**为种子（`argmax w_m`），若并列按（连通域 id, 尺度 id, 坐标）字典序。
  * **跨组件公平**：先把原型分桶（按**连通域 id**），**轮询**从各桶中选取 farthest-first 的赢家，直到达到 `M_max_total`（默认 64）。这样不会让大连通域“吃掉”配额。
* **权重归一**：选中的 `{w_m}` 做 `ŵ_i = w_i / Σ w_i`。可设置最低权重 `w_floor=1e-3` 防止数值过小。

---

## 4) 查询融合与两通道 logits（softmax_τ）

**结论**：思路正确，给出规范的温度缩放与二分类 logits 构造。

* **相似度**：对查询 `F_q(x)` 计算与每个原型的 **余弦相似** `s_i(x)`，与全局原型 `s_g(x)`。
* **温度与融合**：
  `S_fg(x) = β·s_g(x) + Σ_i ŵ_i·s_i(x)`（默认 `β=0.3`），再做 **温度缩放**：
  `z_fg(x) = S_fg(x)/τ`（默认 `τ=0.07`），`z_bg(x) = −z_fg(x)`（对称背景）。
* **两通道 logits**：拼为 `Z(x) = [z_bg(x), z_fg(x)]` 直接喂给上游的 `F.log_softmax(dim=1)` + `NLL/CE` 或 Dice。也可直接输出概率：`P = softmax(Z)`。
* **实现要点**：温度缩放可直接 `Z = torch.stack([-S_fg/τ, S_fg/τ], dim=1)`；训练时别重复 softmax 两次（避免数值损失）。

---

## 5) 多-shot 合并与背景路径（merge & bg）

**结论**：思路正确，补齐合并顺序与共享代码路径。

* **多-shot 合并**：

  1. **每个 shot 独立**：做 CC → 锚点抽样 → 多尺度 → 得到 `{(p,w)}_s` 与 `p_g^s`；
  2. **全局原型**：对各 shot 的前景像素**聚合平均**得 `p_g`（可按每 shot 前景像素数加权）；
  3. **局部原型**：把所有 `{(p,w)}_s` 直接**并集**，再执行第 3 节的 **K-center** 截断至 `M_max_total`。
* **背景处理**：不单独建“背景原型”，而是用对称 logits（`z_bg=-z_fg`）。如需更鲁棒的背景：可加一枚**背景基线偏置** `b_bg`（可学习或常数），实现为 `Z=[-S_fg/τ+b_bg, S_fg/τ]`。初期建议 `b_bg=0` 简洁优先。

---

## 6) 小组件回退与异常情况

* **超小连通域**（`Area_k < τ_area`，如 9 像素）：若窗口统计全部失效，直接**用 `p_g` 替代**该域的局部原型；同时把 `w_m` 设为该域相对面积（避免其影响过大）。
* **全图极低阳性**：当整张支持图阳性像素 `< θ_image`（如 20 px），仅生成 `p_g`，不产生局部原型，仍然可进行匹配（弱化过拟合）。
* **NaN/Inf 容错**：在均值/方差计算前对 mask-sum 加 `eps`；相似度前 `L2` 归一化加 `eps`；K-center 距离上限裁剪到 `[0,2]`。

---

## 7) 模块化接口（`models/am2p.py` 的即插即用骨架）

```python
class AM2P(nn.Module):
    def __init__(self, radii=(4,8,16), alpha=1.2, nmax_comp=8, mmax_total=64,
                 theta_min=8, tau_area=9, beta=0.3, temp=0.07,
                 cc_mode="uf_gpu", anchor_mode="grid"):
        super().__init__()
        self.radii = radii
        self.alpha = alpha
        self.nmax_comp = nmax_comp
        self.mmax_total = mmax_total
        self.theta_min = theta_min
        self.tau_area = tau_area
        self.beta = beta
        self.temp = temp
        self.cc_mode = cc_mode
        self.anchor_mode = anchor_mode

    @torch.no_grad()
    def _connected_components(self, mask):  # HxW bool/byte
        if self.cc_mode == "uf_gpu":
            return union_find_gpu(mask)      # 并查集，返回 HxW 的 int labels
        else:
            return scipy_label_cpu(mask)     # CPU 回退

    @torch.no_grad()
    def _sample_anchors(self, labels, nmax_comp, alpha):
        # 统计每个 label 的 area，计算 n_k；grid or FPS 采样返回 [(y,x,label_id), ...]
        ...

    @torch.no_grad()
    def _local_stats_multi_scale(self, feats, mask, anchors):
        # 积分图或 F.unfold 计算每个 (anchor, r) 的均值/方差/阳性占比
        # 尺度自选，得到 [(p_m, w_m, comp_id, r_id)]
        ...

    @torch.no_grad()
    def _kcenter_prune(self, protos, weights, comp_ids, r_ids):
        # 归一化到球面 -> farthest-first，按 comp 分桶轮询，确定性 seed
        # 返回截断后的 P, W
        ...

    @torch.no_grad()
    def build_prototypes(self, F_s, M_s):
        labels = self._connected_components(M_s)          # CC
        anchors = self._sample_anchors(labels, self.nmax_comp, self.alpha)
        P_raw, W_raw, comp_ids, r_ids = self._local_stats_multi_scale(F_s, M_s, anchors)
        p_g = global_mean(F_s, M_s)                       # 全局
        if len(P_raw) == 0:
            return p_g[None], torch.ones(1, device=F_s.device), p_g  # 只全局
        P, W = self._kcenter_prune(P_raw, W_raw, comp_ids, r_ids)
        W = W / (W.sum() + 1e-6)
        return P, W, p_g

    def forward(self, F_s, M_s, F_q):
        P, W, p_g = self.build_prototypes(F_s, M_s)
        # 余弦相似
        s_local = cosine_sim(F_q, P)          # (B, M, H, W)
        s_global = cosine_sim(F_q, p_g[None]) # (B, 1, H, W)
        S_fg = self.beta * s_global[:,0] + (W[None,:,None,None] * s_local).sum(1)
        # 两通道 logits（温度缩放）
        Z = torch.stack([-S_fg/self.temp, S_fg/self.temp], dim=1)
        return Z
```

---

## 8) 复杂度与默认超参

* **复杂度**：支持端以**积分图**为主，O(HW·|R|)；查询端与 ALP 同阶（M 个原型的点积/余弦）。
* **显存**：远低于 `unfold`，与 ALP 近似；`M_max_total=64` 时对 512×512 的特征图不会成为瓶颈。
* **默认值**：
  `R={4,8,16}`，`α=1.2`，`N_max_comp=8`，`M_max_total=64`，`θ_min=8`，`τ_area=9`，`β=0.3`，`τ=0.07`。
  若前景极稀疏，可把 `β` 提到 `0.5`。

---

## 一句话总结

在不增加任何花哨子网的前提下，**AM²P** 用“**连通域 + 对数面积配额**”和“**积分图驱动的多尺度局部统计**”把 ALP 的原型构建做了更鲁棒的简化；查询侧保持**余弦 + 温度缩放 + 两通道 logits**。这既满足你指出的实现落地点，也保留了 ALP/DINO 体系的最小修改面——如今的 `models/am2p.py` 已成为这套机制的唯一入口。
