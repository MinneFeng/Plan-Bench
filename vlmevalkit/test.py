import torch
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import cv2

def plot_token_attention_matrix(input_ids, attention_matrix, processor, layer_idx, out_dir, max_tokens=180):
    """
    使用纯 Matplotlib 绘制 $S \times S$ 维度的 Token 关联矩阵图，替代 seaborn
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 1. 解码并简化 Token 标签
    raw_tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
    display_tokens = []
    for t in raw_tokens:
        # 简化 Qwen 的特殊占位符
        t_str = t.replace("<|image_pad|>", "<IMG>").replace("<|vision_newline|>", "<V_NL>")
        display_tokens.append(t_str[:15] if len(t_str) > 15 else t_str)

    # 2. 截取注意力矩阵并转换为 float32
    # 注意：这里取序列的前 max_tokens 个 Token 进行可视化
    matrix_to_plot = attention_matrix[:max_tokens, :max_tokens].detach().cpu().float().numpy()
    labels = display_tokens[:max_tokens]

    # 3. 创建画布
    fig, ax = plt.subplots(figsize=(20, 16))

    # 4. 绘制热力图核心 (使用 imshow 替代 sns.heatmap)
    # cmap='viridis' 是你之前喜欢的配色方案
    im = ax.imshow(matrix_to_plot, cmap='viridis', aspect='auto', interpolation='nearest')

    # 5. 设置坐标轴标签
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # 6. 添加颜色条 (Colorbar)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Attention Weight", rotation=-90, va="bottom")

    # 7. 设置标题与布局
    ax.set_title(f"Qwen3-VL Token-to-Token Attention Matrix (Layer {layer_idx})", fontsize=16)
    fig.tight_layout()

    # 8. 保存图像
    save_path = os.path.join(out_dir, f"token_matrix_l{layer_idx}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"原生 Matplotlib 矩阵图已保存至: {save_path}")


def plot_decode_matrix_native(matrix, row_labels, col_labels, layer_idx, out_dir):
    """
    绘制解码阶段生成的 [新Token x 原始Token] 矩阵
    """
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # 转换数据类型并归一化
    matrix_np = matrix.detach().cpu().float().numpy()

    # 原始 Min-Max 归一化
    v_min, v_max = matrix_np.min(), matrix_np.max()
    matrix_norm = (matrix_np - v_min) / (v_max - v_min + 1e-12)

    # 创建画布 (根据 Token 数量动态调整大小)
    fig, ax = plt.subplots(figsize=(max(16, len(col_labels) * 0.2), max(10, len(row_labels) * 0.5)))

    im = ax.imshow(matrix_norm, cmap='viridis', aspect='auto')

    # 设置坐标轴标签
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(row_labels, fontsize=9)

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Attention Weight (Normalized)", rotation=-90, va="bottom")

    ax.set_title(f"Qwen3-VL Decode Stage Attention Matrix (Layer {layer_idx})", fontsize=16)
    fig.tight_layout()

    save_path = os.path.join(out_dir, f"decode_matrix_l{layer_idx}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Decode 矩阵图已保存至: {save_path}")


def run_task_b_only(model, processor, image_path, question, layer_indices, out_dir):
    # 准备输入
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    print("-" * 30)
    print(f"模型生成的答案: {output_text}")
    print("-" * 30)

    # 前向传播提取注意力
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, return_dict=True)

    input_ids = inputs.input_ids[0]

    for l_idx in layer_indices:
        # 提取指定层的注意力张量
        # att 形状为 [num_heads, seq_len, seq_len]
        att = outputs.attentions[l_idx][0]

        # 跨 Head 平均以获得该层的全局视图
        mean_attn = att.mean(dim=0)

        # 将第 0 列（作为被关注对象）置零
        mean_attn[:, 0] = 0
        # 将第 0 行（作为发起关注对象）置零，使矩阵背景更干净
        mean_attn[0, :] = 0

        plot_token_attention_matrix(input_ids, mean_attn, processor, l_idx, out_dir)


def run_decode_matrix_analysis(model, processor, image_path, question, layer_idx, out_dir):
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # 1. 准备输入
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)

    # 2. 生成并捕获注意力 (Decode 阶段)
    with torch.no_grad():
        generate_outputs = model.generate(
            **inputs,
            max_new_tokens=1,  # 观察前 20 个生成的 Token
            output_attentions=True,
            return_dict_in_generate=True,
            do_sample=False
        )

    # 3. 解析标签
    input_ids = inputs.input_ids[0]
    # 横轴：原始 Prompt 的标签
    raw_col_tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
    col_labels = [t.replace("<|image_pad|>", "IMG").replace("<|vision_newline|>", "\\n") for t in raw_col_tokens]

    # 纵轴：新生成的 Token 标签
    prompt_len = input_ids.shape[0]
    gen_ids = generate_outputs.sequences[0][prompt_len:]
    row_labels = [processor.decode([tid]) for tid in gen_ids]

    print(f"生成答案: {''.join(row_labels)}")

    # 4. 提取并拼接解码矩阵
    # generate_outputs.attentions 是 (步数, 层数) 的嵌套结构
    # 每一步的形状是 [batch, heads, 1, current_seq_len]
    all_step_vectors = []

    for step_idx in range(len(generate_outputs.attentions)):
        # 提取指定层的注意力
        step_attn = generate_outputs.attentions[step_idx][layer_idx][0]  # [heads, 1, curr_seq_len]

        # 跨 Head 平均，并只取对原始 Prompt 的注意力 (长度为 prompt_len)
        mean_step_attn = step_attn.mean(dim=0).squeeze(0)[:prompt_len]

        if mean_step_attn.ndim == 2:
            # 取最后一行，且只取前 prompt_len 个列 (即对原始图像+文本的关注)
            final_vec = mean_step_attn[-1, :prompt_len]
        else:
            # 如果已经是 1D，直接截取
            final_vec = mean_step_attn[:prompt_len]

        final_vec = final_vec.clone()
        final_vec[0] = 0

        all_step_vectors.append(final_vec)

    decode_matrix = torch.stack(all_step_vectors)

    plot_decode_matrix_native(decode_matrix, row_labels, col_labels, layer_idx, out_dir)


def run_decode_matrix_and_overlay(model, processor, image_path, question, layer_idx, out_dir):
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # 1. 准备输入
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(
        model.device)

    # 2. 生成并捕获注意力 (Decode 阶段)
    # 注意：这里 max_new_tokens=1，意味着我们只看生成的第一个 Token 的注意力
    with torch.no_grad():
        generate_outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            output_attentions=True,
            return_dict_in_generate=True,
            do_sample=False
        )

    # 3. 解析标签与基础信息
    input_ids = inputs.input_ids[0]
    prompt_len = input_ids.shape[0]
    gen_ids = generate_outputs.sequences[0][prompt_len:]
    row_labels = [processor.decode([tid]).strip() for tid in gen_ids]
    generated_token_str = row_labels[0] if row_labels else "unknown"
    print(f"生成答案 Token: '{generated_token_str}'")

    # 获取图像原始信息用于后续绘图
    raw_image = Image.open(image_path).convert("RGB")
    _, grid_h, grid_w = inputs.image_grid_thw[0]

    # --- 准备视觉掩码 (用于剔除文本和换行符) ---
    v_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    v_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    v_nl_id = 151655  # 视觉换行符 ID

    v_start = (input_ids == v_start_id).nonzero()[0][0].item() + 1
    v_end = (input_ids == v_end_id).nonzero()[0][0].item()

    # 创建一个布尔掩码，标记 prompt 中哪些位置是有效的视觉 Token
    vision_range_mask = torch.zeros(prompt_len, dtype=torch.bool, device=model.device)
    vision_range_mask[v_start:v_end] = True
    is_newline = (input_ids[:prompt_len] == v_nl_id)
    # 最终掩码：在视觉范围内 且 不是换行符
    final_vision_mask = vision_range_mask & (~is_newline)

    # 4. 提取解码向量
    all_step_vectors = []
    for step_idx in range(len(generate_outputs.attentions)):
        step_attn = generate_outputs.attentions[step_idx][layer_idx][0]
        # 提取对 Prompt 的注意力并跨 Head 平均
        mean_step_attn = step_attn.mean(dim=0).squeeze(0)[:prompt_len]

        if mean_step_attn.ndim == 2:
            # 处理可能的 2D 情况，取最后一行
            final_vec = mean_step_attn[-1, :prompt_len]
        else:
            final_vec = mean_step_attn[:prompt_len]

        # 剔除沉降点
        final_vec = final_vec.clone()
        final_vec[0] = 0
        all_step_vectors.append(final_vec)

    print("正在生成 2D 叠加热力图...")

    # 由于 max_new_tokens=1，我们取第一个生成的向量
    target_vec = all_step_vectors[0]

    # 调用辅助函数进行投影
    heatmap_grid = project_decode_attn_to_image(
        target_vec, final_vision_mask, grid_h, grid_w, raw_image.size
    )

    # 应用色图并叠加
    heatmap_img = cv2.applyColorMap(np.uint8(255 * heatmap_grid), cv2.COLORMAP_JET)
    heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
    # alpha 0.6 原图, 0.4 热力图
    overlay = cv2.addWeighted(np.array(raw_image), 0.6, heatmap_img, 0.4, 0)

    # 保存图片
    # 文件名包含生成的 Token 内容，方便识别
    safe_token_str = "".join([c for c in generated_token_str if c.isalnum()]) or "symbol"
    save_path = os.path.join(out_dir, f"decode_overlay_layer{layer_idx}_token_{safe_token_str}.png")
    Image.fromarray(overlay).save(save_path)
    print(f"2D 叠加热力图已保存至: {save_path}")


def project_decode_attn_to_image(attn_vec, vision_mask, grid_h, grid_w, target_res):

    # 1. 应用掩码提取纯视觉 Token 的注意力，并转为 float32
    vec = attn_vec[vision_mask].detach().cpu().float()

    actual_size = vec.numel()
    if actual_size == 0:
        print("Warning: 未检测到视觉注意力信号，返回全黑图。")
        return np.zeros((target_res[1], target_res[0]), dtype=np.float32)

    expected_size = grid_h * grid_w

    # 2. 自动修复 Qwen3-VL 的 2x2 Pooling 导致的尺寸不匹配
    # Qwen 会将 2x2 的 patch 压缩为一个 token，导致 actual_size 通常是 expected_size 的 1/4
    if actual_size != expected_size:
        scale = math.sqrt(actual_size / expected_size)
        new_h, new_w = int(grid_h * scale), int(grid_w * scale)

        # 鲁棒性调整：确保乘积精确等于 actual_size
        if new_h * new_w != actual_size:
            new_w = grid_w // 2 if actual_size % (grid_w // 2) == 0 else int(math.sqrt(actual_size))
            new_h = actual_size // new_w

        # print(f"DEBUG: Pooling detected. Resizing grid from {grid_h}x{grid_w} to {new_h}x{new_w}")
        grid_h, grid_w = new_h, new_w

    # 3. 原始 Min-Max 归一化 (展示最真实的分布状态)
    v_min, v_max = vec.min(), vec.max()
    denom = (v_max - v_min).clamp(min=1e-12)
    vec_norm = (vec - v_min) / denom

    # 4. 重塑为二维网格并拉伸回原图尺寸
    grid = vec_norm.reshape(grid_h, grid_w).numpy()
    # 使用最近邻插值保持块状结构，或者使用双线性插值获得更平滑的效果
    return cv2.resize(grid, target_res, interpolation=cv2.INTER_NEAREST)


# --- 3. 模型加载 ---
def load_model(path="/defaultShare/Qwen3-VL-8B-Instruct"):
    processor = AutoProcessor.from_pretrained(path)
    # 必须使用 eager 模式以支持注意力导出
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    ).eval()
    model.config.output_attentions = True
    return model, processor


# --- 调用执行 ---
if __name__ == "__main__":
    path = "/defaultShare/Qwen3-VL-8B-Instruct"
    q_img = "/root/LMUData/fjt/VLMEvalKit/resource/ourbench/questions/1842.jpg"
    q_txt = "Read the question and options shown in the image(s). Answer with only the single letter of the correct option (e.g., A, B, C, D)."

    qwen_model, processor = load_model(path)
    '''run_task_b_only(
        model=qwen_model,
        processor=processor,
        image_path=q_img,
        question=q_txt,
        layer_indices=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36],  # 重点分析语义对齐较强的深层
        out_dir="/root/LMUData/fjt/VLMEvalKit/output_headmaps"
    )'''
    run_decode_matrix_and_overlay(
        model=qwen_model,
        processor=processor,
        image_path=q_img,
        question=q_txt,
        layer_idx=33,  # 推荐 32 或 33 层
        out_dir="/root/LMUData/fjt/VLMEvalKit/output_headmaps"
    )