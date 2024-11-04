import torch
from nnsight import LanguageModel
from sae_lens import SAE
from datasets import load_dataset
from transformer_lens import utils

from window_data import save_window_data, upload_chunks_to_hub

torch.set_grad_enabled(False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16


# -----  Set these -----
layer = 13
hook_pt = f"blocks.{layer}.hook_resid_post"
n_ctx = 1024
stem_length = 64
num_contexts = 10683
batch_size = 16
file_prefix = f"gemma_2_2b_layer{layer}_contexts{num_contexts}_stem{stem_length}"
# ----------------------


# Load gemma model and dataset
model = LanguageModel("google/gemma-2-2b", device_map=device, torch_dtype=dtype)
submodule = model.model.layers[layer] # nnsight will grab activations at the OUTPUT of this submodule

# load gemma scope SAE
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gemma-scope-2b-pt-res-canonical",
    sae_id = f"layer_{layer}/width_16k/canonical",
)
sae = sae.to(device).to(dtype)

# Load OpenWebText dataset
eval_data = load_dataset("stas/openwebtext-10k", split="train")
tokenized_data = utils.tokenize_and_concatenate(eval_data, model.tokenizer, max_length=n_ctx)
tokenized_data = tokenized_data.shuffle(42)
eval_tokens = tokenized_data["tokens"]


save_window_data(
    model,
    submodule,
    sae,
    eval_tokens[:, :stem_length],
    save_dir = "/root/features-over-time/data",
    num_contexts = num_contexts, #eval_tokens.shape[0],
    batch_size = batch_size,
    window_length = 64,
    stride = 16,
    num_tokens_to_generate = n_ctx - stem_length,
    temperature = 1.0,
    return_feature_acts = False,
    compress = True
)


# Upload chunks
drive_metadata = upload_chunks_to_hub(
    metadata_file="/root/features-over-time/data/window_data_metadata.json",
    repo_id = "jacobcd52/features-over-time",
    file_prefix = file_prefix,
    chunk_size_mb=500,
    token="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"
)