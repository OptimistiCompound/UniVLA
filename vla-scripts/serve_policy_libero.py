from typing import Optional, Sequence
import os
import time
import json
import torch
import torch.nn as nn
import cv2 as cv
import numpy as np
from PIL import Image

# openpi imports
# import flax
# from flax.traverse_util import flatten_dict
import dataclasses
import enum
import socket
import tyro
import logging
from univla_client.base_policy import BasePolicy
from serving import websocket_policy_server


from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize UniVLA model
def get_vla(pretrained_checkpoint: str):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vla = AutoModelForVision2Seq.from_pretrained(
        pretrained_checkpoint,
        # attn_implementation="flash_attention_2",
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        load_in_4bit=False,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla


def get_processor(pretrained_checkpoint: str):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(pretrained_checkpoint, trust_remote_code=True)
    return processor

from prismatic.models.policy.transformer_utils import MAPBlock


class ActionDecoderHead(torch.nn.Module):
    def __init__(self, window_size = 12, hidden_dim = 512):
        super().__init__()
        self.latent_action_pool = MAPBlock(n_latents = 1, vis_dim = 4096, embed_dim = hidden_dim, n_heads = hidden_dim // 64)
        self.visual_pool = MAPBlock(n_latents = 1, vis_dim = 4096, embed_dim = hidden_dim, n_heads = hidden_dim // 64)

        self.proj = nn.Sequential(
                                nn.Linear(hidden_dim, 7 * window_size), # 512, 84
                                nn.Tanh(),
                    )


    def forward(self, latent_action_tokens, visual_embed, proprio=None):
        visual_embed = self.visual_pool(visual_embed)
        latent_action_tokens = latent_action_tokens[:, -4:]
        action_token = self.latent_action_pool(latent_action_tokens, init_embed = visual_embed)

        action = self.proj(action_token)

        return action


class ActionDecoder(nn.Module):
    def __init__(self,window_size=12):
        super().__init__()
        self.net = ActionDecoderHead(window_size=window_size)
        self.window_size = window_size
        self.temporal_size = window_size
        self.temporal_size = 8
        self.temporal_mask = torch.flip(torch.triu(torch.ones(self.temporal_size, self.temporal_size, dtype=torch.bool)), dims=[1]).numpy()
        
        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)

        # Action chunking with temporal aggregation
        balancing_factor = 0.1
        self.temporal_weights = np.array([np.exp(-1 * balancing_factor * i) for i in range(self.temporal_size)])[:, None]


    def reset(self):
        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)

    
    def forward(self, latent_actions, visual_embed, proprio=None):
        # Forward action decoder
        # NOTE: We take the last 8 actions in an action chunk for non-blocking controller to tackle possible mismatch led by model latency
        pred_action = self.net(latent_actions.to(torch.float), visual_embed.to(torch.float), proprio).reshape(-1, self.window_size, 7)[:, self.window_size - self.temporal_size:]
        pred_action = np.array(pred_action.tolist())
        
        # Shift action buffer
        self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
        self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
        self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
        self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
        self.action_buffer_mask = self.action_buffer_mask * self.temporal_mask

        # Add to action buffer
        self.action_buffer[0] = pred_action  
        self.action_buffer_mask[0] = np.array([True] * self.temporal_mask.shape[0], dtype=np.bool_)

        # Ensemble temporally to predict action
        action_prediction = np.sum(self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1] * self.temporal_weights, axis=0) / np.sum(self.action_buffer_mask[:, 0:1] * self.temporal_weights)


        return action_prediction


class UniVLAPolicy(BasePolicy):
    def __init__(
        self,
        saved_model_path: str = "/home/jhwang/.cache/huggingface/hub/models--qwbu--univla-7b-224-sft-libero/snapshots/0d1979ebd1b7ad00d6a8a02a98213a0835dc82a1/univla-libero-spatial",
        decoder_path: str = "/home/jhwang/.cache/huggingface/hub/models--qwbu--univla-7b-224-sft-libero/snapshots/0d1979ebd1b7ad00d6a8a02a98213a0835dc82a1/univla-libero-spatial/action_decoder.pt",
        unnorm_key: Optional[str] = None,
        horizon: int = 1,
        pred_action_horizon: int = 8,
        exec_horizon: int = 1,
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # --- 自动下载逻辑 ---
        # 1. 定义 Repo ID 和 子文件夹
        repo_id = saved_model_path
        subfolder = "univla-libero-spatial" # 这是一个硬编码的子文件夹

        # 2. 判断是否已经是本地路径
        if os.path.isdir(saved_model_path):
            print(f"[*] Using local model path: {saved_model_path}")
            model_checkpoint_path = saved_model_path
            decoder_checkpoint_path = decoder_path
        else:
            from huggingface_hub import snapshot_download, hf_hub_download
            print(f"[*] Downloading model from HF Repo: {repo_id}, Subfolder: {subfolder}...")
            try:
                # 下载模型文件（只下载 subfolder 下的内容）snapshot_download 返回的是缓存根目录
                snapshot_root = snapshot_download(repo_id=repo_id, allow_patterns=[f"{subfolder}/*"])
                model_checkpoint_path = os.path.join(snapshot_root, subfolder)
                
                # 下载 action_decoder.pt
                print(f"[*] Downloading action_decoder.pt...")
                try:
                    # 尝试在 subfolder 里找
                    decoder_checkpoint_path = hf_hub_download(repo_id=repo_id, filename="action_decoder.pt", subfolder=subfolder)
                except Exception:
                    # 如果不在 subfolder，尝试在根目录找
                    decoder_checkpoint_path = hf_hub_download(repo_id=repo_id, filename="action_decoder.pt")
            
            except Exception as e:
                print(f"Error downloading from HF: {e}")
                raise e
        print(f"[*] Resolved Local Model Path: {model_checkpoint_path}")
        print(f"[*] Resolved Local Decoder Path: {decoder_checkpoint_path}")
        # ---------------

        # Load model
        self.vla = get_vla(model_checkpoint_path).cuda()
        self.processor = get_processor(model_checkpoint_path)

        # Load action decoder
        self.action_decoder = ActionDecoder(window_size = pred_action_horizon)
        
        # debug
        state_dict = torch.load(decoder_checkpoint_path)
        print(state_dict.keys())

        self.action_decoder.net.load_state_dict(state_dict)
        self.action_decoder.eval().cuda()


        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.exec_horizon = exec_horizon

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.unnorm_key = unnorm_key

        self.task = None
        self.task_description = None
        self.num_image_history = 0

        self.prev_hist_action = ['']

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.action_decoder.reset()
        self.prev_hist_action = ['']

    def _process_image(self, image):
        if isinstance(image, np.ndarray):
            image = image.squeeze()
            # 如果是 (3, H, W) 格式，转置为 (H, W, 3)；如果是 (H, W, 3) 则保持不变
            if image.ndim == 3 and image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            # 如果是浮点数 (0-1)，缩放到 0-255
            if image.dtype.kind == 'f':
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        elif isinstance(image, torch.Tensor):
            image = (image.squeeze().permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
        
        return image

    def infer(
        self, obs: dict
    ) -> dict:
        """
        Input:
            image: torch.Tensor of shape (3, H, W)
            task_description: str; task description
            proprio: torch.Tensor; proprioceptive state of the robot
        Output:
            action: numpy.array of shape (1, 7); processed action to be sent to the robot arms
        """
        raw_image        = obs["full_image"]
        task_description = obs["instruction"]
        proprio          = obs["state"]

        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        image: Image.Image = Image.fromarray(self._process_image(raw_image))

        if len(self.prev_hist_action[-1]) > 0:
            prompt = f"In: What action should the robot take to {task_description.lower()}? History action {self.prev_hist_action[-1]}\nOut:"
        else:
            prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"


        # predict action (7-dof; un-normalize for bridgev2)
        inputs = self.processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
        latent_action, visual_embed, generated_ids = self.vla.predict_latent_action(**inputs, unnorm_key=self.unnorm_key, do_sample=True, temperature=0.75, top_p = 0.9)

        latent_action_detokenize = [f'<ACT_{i}>' for i in range(32)]
        hist_action = ''
        for latent_action_ids in generated_ids[0]:
            hist_action += latent_action_detokenize[latent_action_ids.item() - 32001]
        self.prev_hist_action.append(hist_action)
        action = self.action_decoder(latent_action, visual_embed, proprio) # tuple[dict[str, np.ndarray], dict[str, np.ndarray]]
        
        action_chunk = [action] # to conform with sifer action format
        action = {"actions": action_chunk}

        return action



# class PolicyRecorder(BasePolicy):
#     """Records the policy's behavior to disk."""
#     from pathlib import Path
#     def __init__(self, policy: BasePolicy, record_dir: str):
#         self._policy = policy

#         logging.info(f"Dumping policy records to: {record_dir}")
#         self._record_dir = Path(record_dir)
#         self._record_dir.mkdir(parents=True, exist_ok=True)
#         self._record_step = 0

#     def infer(self, obs: dict) -> dict:  # type: ignore[misc]
#         results = self._policy.infer(obs)
#         data = {"inputs": obs, "outputs": results}
#         data = flatten_dict(data, sep="/")
#         output_path = self._record_dir / f"step_{self._record_step}"
#         self._record_step += 1

#         np.save(output_path, np.asarray(data))
#         return results


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""
    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None
    # Port to serve the policy on.
    port: int = 8010
    # Record the policy's behavior for debugging.
    record: bool = False
    
    saved_model_path: str = "/home/jhwang/.cache/huggingface/hub/models--qwbu--univla-7b-224-sft-libero/snapshots/0d1979ebd1b7ad00d6a8a02a98213a0835dc82a1/univla-libero-spatial"
    decoder_path: str = "/home/jhwang/.cache/huggingface/hub/models--qwbu--univla-7b-224-sft-libero/snapshots/0d1979ebd1b7ad00d6a8a02a98213a0835dc82a1/univla-libero-spatial/action_decoder.pt"
    unnorm_key: str = "libero_spatial_no_noops"
    horizon: int = 1
    pred_action_horizon: int = 12
    exec_horizon: int = 1
    action_scale: float = 1.0

def main(args: Args) -> None:
    policy = UniVLAPolicy(
        saved_model_path=args.saved_model_path,
        decoder_path=args.decoder_path,
        unnorm_key=args.unnorm_key,
        horizon=args.horizon,
        pred_action_horizon=args.pred_action_horizon,
        exec_horizon=args.exec_horizon,
        action_scale=args.action_scale
        )

    # Record the policy's behavior and save it to .npy files
    # if args.record:
    #     policy = PolicyRecorder(policy, "policy_records")
    # logging
    logging.info("Creating server (host: 0.0.0.0, ip: %s)", args.port)

    # create server
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=None,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
