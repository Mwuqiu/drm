import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import shutil
import io
import json
import base64
from typing import List, Dict, Any
from openai import OpenAI

import yaml
from pathlib import Path

def load_task_config(task_name: str) -> dict:
    config_path = Path("task_configs") / f"{task_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found for task: {task_name}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def encode_image_to_data_url(png_path: str) -> str:
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def build_messages(
    episode_id: int,
    frame_infos: List[Dict[str, Any]],
    task_config: dict,
) -> List[Dict[str, Any]]:
    
    system_prompt_template = task_config['system_prompt']
    task_description = task_config['task_description']
    schema_text = task_config['schema_text']
    evaluation_criteria = task_config['evaluation_criteria']
    success_examples = task_config['success_examples']
    failure_examples = task_config['failure_examples']

    system_prompt = system_prompt_template.format(task_description=task_description)

    few_shot_content = []

    for example in success_examples:
        few_shot_content.append({"type": "text", "text": f"=== {example['title']} ==="})
        few_shot_content.append({"type": "text", "text": example['explanation']})
        for info in example['frames']:
            few_shot_content.append({"type": "text", "text": f"Example Frame {info['frame_idx']}"})
            main_data_url = encode_image_to_data_url(info["paths"]["main"])
            few_shot_content.append({"type": "text", "text": "Main View:"})
            few_shot_content.append({"type": "image_url", "image_url": {"url": main_data_url}})
            # wrist_data_url = encode_image_to_data_url(info["paths"]["wrist"])
            # few_shot_content.append({"type": "text", "text": "Wrist View:"})
            # few_shot_content.append({"type": "image_url", "image_url": {"url": wrist_data_url}})

    for example in failure_examples:
        few_shot_content.append({"type": "text", "text": f"=== {example['title']} ==="})
        few_shot_content.append({"type": "text", "text": example['explanation']})
        for info in example['frames']:
            few_shot_content.append({"type": "text", "text": f"Example Frame {info['frame_idx']}"})
            main_data_url = encode_image_to_data_url(info["paths"]["main"])
            few_shot_content.append({"type": "text", "text": "Main View:"})
            few_shot_content.append({"type": "image_url", "image_url": {"url": main_data_url}})
            # wrist_data_url = encode_image_to_data_url(info["paths"]["wrist"])
            # few_shot_content.append({"type": "text", "text": "Wrist View:"})
            # few_shot_content.append({"type": "image_url", "image_url": {"url": wrist_data_url}})

    user_content: List[Dict[str, Any]] = []
    user_content.extend(few_shot_content)
    user_content.append({"type": "text", "text": f"\n=== Episode to Evaluate: {episode_id} ==="})
    user_content.append({"type": "text", "text": "Evaluation Criteria:"})
    user_content.append({"type": "text", "text": evaluation_criteria})
    user_content.append({"type": "text", "text": schema_text.strip()})
    user_content.append({"type": "text", "text": "The following are the keyframes in chronological order:"})

    for info in frame_infos:
        frame_idx = info["frame_idx"]
        paths = info["paths"] 
        tag = "(Key Event)" if info.get("is_key_event", False) else ""
        aux = info.get("aux")
        
        user_content.append({"type": "text", "text": f"--- Frame {frame_idx} {tag} ---"})
        if aux:
            user_content.append({"type": "text", "text": f"Auxiliary Info: {json.dumps(aux, ensure_ascii=False)}"})
            
        main_data_url = encode_image_to_data_url(paths["main"])
        user_content.append({"type": "text", "text": "Main View:"})
        user_content.append({
            "type": "image_url",
            "image_url": {"url": main_data_url}
        })
        
        # wrist_data_url = encode_image_to_data_url(paths["wrist"])
        # user_content.append({"type": "text", "text": "Wrist View:"})
        # user_content.append({
        #     "type": "image_url",
        #     "image_url": {"url": wrist_data_url}
        # })
        
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    return messages

def call_gpt_label(messages: List[Dict[str, Any]], model: str = "gpt-4o") -> Dict[str, Any]:
    import os
    for key in ["ALL_PROXY", "all_proxy"]:
        os.environ.pop(key, None)

    client = OpenAI(
        api_key="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",
    )

    # client = OpenAI(
    #     api_key="sk-FRXl1ihtpO0eTyFAv58CSZEdmn6UVNf3O2KQD9z9QHt2cyoR", 
    #     base_url="https://api.openai-proxy.org/v1",
    # )

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    return json.loads(content)

def label_episode(
    episode_id: int,
    episode_dir: str,
    sampled_indices: List[int],
    detected_subgoals: List[int],
    actions_for_frames: Dict[int, list] = None
) -> Dict[str, Any]:
    frame_infos = []
    for idx in sampled_indices:
        base_filename = f"frame_{idx:04d}{'_key_event' if idx in detected_subgoals else ''}"
        filename_main = f"{base_filename}_main.png"
        filename_wrist = f"{base_filename}_wrist.png"

        path_main = os.path.join(episode_dir, filename_main)
        path_wrist = os.path.join(episode_dir, filename_wrist)

        aux = None
        if actions_for_frames and idx in actions_for_frames:
            aux = {"action": actions_for_frames[idx]}
        
        frame_infos.append({
            "frame_idx": idx,
            "paths": {"main": path_main, "wrist": path_wrist},
            "is_key_event": (idx in detected_subgoals),
            "aux": aux
        })

    # current_task_name = "pick_up_the_ketchup_and_place_it_in_the_basket"  # task-0
    # current_task_name = "pick_up_the_cream_cheese_and_put_it_in_the_tray" # task-1
    # current_task_name = "turn_on_the_stove_and_put_the_frying_pan_on_it" # task-3
    # current_task_name = "pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy" # task-4
    current_task_name = "pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy" # task-5
    task_config = load_task_config(current_task_name)
    messages = build_messages(episode_id, frame_infos,task_config=task_config)
    result = call_gpt_label(messages, model="gpt-4o")
    out_path = os.path.join(episode_dir, f"episode_{episode_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result

if __name__ == "__main__":
    REPO_ID = "libero_elegant/task_6"
    EPISODE_INDEX = 1
    INTERVAL_THRESHOLD = 20
    OUTPUT_DIR = "sampled_frames_output/task_6"

    from reward_dataset import RewardLeRobotDataset
    dataset = RewardLeRobotDataset(repo_id=REPO_ID)
    ep_start = dataset.episode_data_index["from"][EPISODE_INDEX].item()
    ep_end = dataset.episode_data_index["to"][EPISODE_INDEX].item()
    ep_len = ep_end - ep_start

    episode_data = dataset.hf_dataset[ep_start:ep_end]
    ep_states = torch.stack(episode_data["observation.state"])
    ep_actions = torch.stack(episode_data["action"])

    print("-" * 50)
    print(f"测试 Episode: {EPISODE_INDEX}, 总帧数: {ep_len}")
    print("-" * 50)

    sampled_indices, detected_subgoals = dataset.get_sampling_indices_with_subgoals(
        ep_states, ep_actions, ep_len, interval_threshold=INTERVAL_THRESHOLD
    )

    print(f"检测到的子目标（关键事件）帧: \n{detected_subgoals}")
    print(f"\n最终采样总帧数: {len(sampled_indices)}")
    print(f"最终采样的帧索引: \n{sampled_indices}")

    if sampled_indices:
        episode_output_dir = os.path.join(OUTPUT_DIR, f"episode_{EPISODE_INDEX}")

        if os.path.exists(episode_output_dir):
            shutil.rmtree(episode_output_dir)

        os.makedirs(episode_output_dir)

        num_sampled = len(sampled_indices)
        
        cols_per_pair = 2 
        pairs_per_row = 5
        cols = pairs_per_row * cols_per_pair
        rows = int(np.ceil(num_sampled / pairs_per_row))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2))
        if num_sampled == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten()

        fig.suptitle(f'Sampled Frames for Episode {EPISODE_INDEX} (Main / Wrist)', fontsize=20)

        for i, frame_idx in enumerate(sampled_indices):
            image_tensor_main = episode_data["observation.images.image"][frame_idx]
            image_np_main = image_tensor_main.permute(1, 2, 0).numpy()

            image_tensor_wrist = episode_data["observation.images.wrist_image"][frame_idx]
            image_np_wrist = image_tensor_wrist.permute(1, 2, 0).numpy()

            ax_main = axes_flat[i * cols_per_pair]
            ax_main.imshow(image_np_main)
            title_main = f"Frame: {frame_idx} (Main)"
            color = 'black'
            if frame_idx in detected_subgoals:
                title_main += "\n(Key Event)"
                color = 'red'
            ax_main.set_title(title_main, fontsize=10, color=color)
            ax_main.axis('off')
            
            ax_wrist = axes_flat[i * cols_per_pair + 1]
            ax_wrist.imshow(image_np_wrist)
            title_wrist = f"Frame: {frame_idx} (Wrist)"
            ax_wrist.set_title(title_wrist, fontsize=10, color='black')
            ax_wrist.axis('off')

            base_filename = f"frame_{frame_idx:04d}{'_key_event' if frame_idx in detected_subgoals else ''}"
            filename_main = f"{base_filename}_main.png"
            filename_wrist = f"{base_filename}_wrist.png"
            
            filepath_main = os.path.join(episode_output_dir, filename_main)
            filepath_wrist = os.path.join(episode_output_dir, filename_wrist)

            pil_img_main = Image.fromarray((image_np_main * 255).astype(np.uint8))
            pil_img_main.save(filepath_main)

            pil_img_wrist = Image.fromarray((image_np_wrist * 255).astype(np.uint8))
            pil_img_wrist.save(filepath_wrist)

        for i in range(num_sampled * cols_per_pair, len(axes_flat)):
            axes_flat[i].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show() # 如果需要，可以取消注释以显示图像

        # actions_for_frames = None

        # result = label_episode(
        #     episode_id=EPISODE_INDEX,
        #     episode_dir=episode_output_dir,
        #     sampled_indices=sampled_indices,
        #     detected_subgoals=detected_subgoals,
        #     actions_for_frames=actions_for_frames
        # )
        # print("模型评测结果：", json.dumps(result, ensure_ascii=False, indent=2))