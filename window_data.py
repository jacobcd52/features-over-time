from pathlib import Path
import h5py
import torch
from tqdm import tqdm
import numpy as np
import json
from typing import Optional, Tuple, Union, List, NamedTuple

import os
import json
import tarfile
import shutil
from pathlib import Path
from typing import Union
from tqdm import tqdm
from huggingface_hub import HfApi, upload_file, hf_hub_download

import os
import json
import tarfile
import shutil
from pathlib import Path
from typing import Union
from tqdm import tqdm
from huggingface_hub import HfApi, upload_file, hf_hub_download


#@title Function to get a batch of window data
class WindowDataOutputs(NamedTuple):
    window_averages: torch.Tensor
    window_p_active: torch.Tensor
    feature_acts: Optional[torch.Tensor] = None
    tokens_plus_generation: Optional[torch.Tensor] = None

def get_batch_window_data(
        model,
        submodule,
        sae,
        batch_tokens,
        window_length=64,
        stride=16,
        return_feature_acts=False,
        num_tokens_to_generate=0,
        temperature=None
        ):
    """
    Run a model forward pass using nnsight, extract activations, and process them through an SAE.

    Args:
        model: An nnsight LanguageModel instance
        sae: Trained Sparse Autoencoder
        batch_tokens: Input tokens to process (assumed to be a tensor of token ids)
        layer_name: Name of the layer to extract activations from (e.g., "transformer.h.5")
        window_length: Size of sliding window
        stride: Step size for sliding window
        return_feature_acts: Whether to return raw feature activations
        num_tokens_to_generate: Number of tokens to generate after the input tokens
        temperature: Sampling temperature for generation
    """
    assert isinstance(batch_tokens, torch.Tensor)

    feature_acts = None
    tokens_plus_generation = None

    if num_tokens_to_generate > 0:
        with model.generate(batch_tokens,
                            min_new_tokens=num_tokens_to_generate,
                            max_new_tokens=num_tokens_to_generate,
                            do_sample=temperature>0,
                            temperature=temperature,
                            pad_token_id=model.tokenizer.eos_token_id
                            ) as tracer:
            tokens_plus_generation = model.generator.output.save()

            init_acts = submodule.output[0]
            generated_acts = torch.cat([submodule.next().output[0] for _ in range(num_tokens_to_generate-1)], dim=1)
            acts = torch.cat([init_acts, generated_acts], dim=1)

            feature_acts_temp = sae.encode(acts)
            windows = feature_acts_temp.unfold(1, window_length, stride)
            window_averages = windows.mean(dim=-1).save()
            window_p_active = (windows > 0).float().mean(dim=-1).save()
            if return_feature_acts:
                feature_acts = feature_acts_temp.save()
    else:
        with model.trace() as tracer:
            with tracer.invoke(batch_tokens):
                acts = submodule.output[0]
                feature_acts_temp = sae.encode(acts).save() if return_feature_acts else sae.encode(acts)
                windows = feature_acts_temp.unfold(1, window_length, stride)
                window_averages = windows.mean(dim=-1).save()
                window_p_active = (windows > 0).float().mean(dim=-1).save()
                if return_feature_acts:
                    feature_acts = feature_acts_temp.save()

    # Convert to sparse format and move to CPU
    window_averages_sparse = window_averages.to_sparse_coo().cpu()
    window_p_active_sparse = window_p_active.to_sparse_coo().cpu()

    return WindowDataOutputs(window_averages_sparse, window_p_active_sparse, feature_acts, tokens_plus_generation)




#@title Function to store a small amount of window data on CPU (useful for dashboards)
def get_window_data_cpu(
        model,
        submodule,
        sae, tokens,
        num_contexts=1_000,
        batch_size=4,
        window_length=64,
        stride=16,
        return_feature_acts=False,
        num_tokens_to_generate=0,
        temperature=None
        ):
    window_averages = []
    window_p_active = []
    feature_acts = []
    tokens_plus_generation = []

    for b in tqdm(range(num_contexts // batch_size)):
        batch_tokens = tokens[b*batch_size:(b+1)*batch_size]
        out = get_batch_window_data(
            model,
            submodule,
            sae,
            batch_tokens,
            window_length=window_length,
            stride=stride,
            return_feature_acts=return_feature_acts,
            num_tokens_to_generate=num_tokens_to_generate,
            temperature=temperature
            )
        window_averages.append(out.window_averages)
        window_p_active.append(out.window_p_active)
        feature_acts.append(out.feature_acts)
        tokens_plus_generation.append(out.tokens_plus_generation)

    window_averages = torch.cat(window_averages, dim=0)
    window_p_active = torch.cat(window_p_active, dim=0)
    feature_acts = torch.cat(feature_acts, dim=0) if return_feature_acts else None
    tokens_plus_generation = torch.cat(tokens_plus_generation, dim=0) if num_tokens_to_generate > 0 else None

    return WindowDataOutputs(window_averages, window_p_active, feature_acts, tokens_plus_generation)




#@title Function to save window data to disk
def save_window_data(
    model,
    submodule,
    sae,
    tokens,
    save_dir: Union[str, Path],
    save_prefix: str = "window_data",
    num_contexts: int = 1_000,
    batch_size: int = 16,
    window_length: int = 64,
    stride: int = 16,
    num_tokens_to_generate: int = 0,
    temperature: Optional[float] = None,
    return_feature_acts: bool = False,
    compress: bool = True
) -> Path:
    """
    Process and save window activations and tokens to multiple files, managing memory.
    Returns path to metadata file.
    """
    if num_tokens_to_generate > 0:
        assert temperature is not None, "Temperature must be provided for generation"

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = save_dir / "chunks"
    chunk_dir.mkdir(exist_ok=True)

    total_batches = num_contexts // batch_size
    compression = 'gzip' if compress else None

    # Calculate window positions for token mapping
    seq_length = tokens.size(1)
    num_windows = (seq_length - window_length) // stride + 1
    window_starts = torch.arange(0, num_windows) * stride
    window_ends = window_starts + window_length

    # Save window mapping info separately
    mapping_file = save_dir / f"{save_prefix}_mapping.h5"
    with h5py.File(mapping_file, 'w') as f:
        f.create_dataset('window_starts', data=window_starts.numpy(), compression=compression)
        f.create_dataset('window_ends', data=window_ends.numpy(), compression=compression)
        f.attrs['window_length'] = window_length
        f.attrs['stride'] = stride
        f.attrs['seq_length'] = seq_length
        f.attrs['num_tokens_generated'] = num_tokens_to_generate
        f.attrs['total_batches'] = total_batches

    # Process and save batches individually
    metadata = {
        'mapping_file': str(mapping_file),
        'batch_files': [],
        'token_files': []
    }

    for b in tqdm(range(total_batches), desc="Processing batches"):
        # Get current batch of tokens
        batch_tokens = tokens[b*batch_size:(b+1)*batch_size]

        # Process current batch
        batch_result = get_batch_window_data(
            model, submodule, sae, batch_tokens,
            window_length=window_length,
            stride=stride,
            return_feature_acts=return_feature_acts,
            num_tokens_to_generate=num_tokens_to_generate,
            temperature=temperature
        )

        # Save batch data
        batch_file = chunk_dir / f"{save_prefix}_batch_{b}.h5"
        with h5py.File(batch_file, 'w') as f:
            avg = batch_result[0]
            p_act = batch_result[1]

            # Save window averages
            avg_grp = f.create_group('window_averages')
            avg_grp.create_dataset('indices', data=avg.indices().detach().numpy(), compression=compression)
            avg_grp.create_dataset('values', data=avg.values().float().detach().numpy(), compression=compression)
            avg_grp.attrs['shape'] = avg.size()

            # Save activation probabilities
            p_act_grp = f.create_group('window_p_active')
            p_act_grp.create_dataset('indices', data=p_act.indices().detach().numpy(), compression=compression)
            p_act_grp.create_dataset('values', data=p_act.values().float().detach().numpy(), compression=compression)
            p_act_grp.attrs['shape'] = p_act.size()

            if return_feature_acts and batch_result[2] is not None:
                feat_acts = batch_result[2]
                feat_grp = f.create_group('feature_acts')
                feat_grp.create_dataset('indices', data=feat_acts.indices().numpy(), compression=compression)
                feat_grp.create_dataset('values', data=feat_acts.values().float().numpy(), compression=compression)
                feat_grp.attrs['shape'] = feat_acts.size()

            if num_tokens_to_generate > 0 and batch_result[3] is not None:
                f.create_dataset('tokens_plus_generation',
                               data=batch_result[3].cpu().numpy(),
                               compression=compression)

        # Save batch tokens separately
        token_file = chunk_dir / f"{save_prefix}_tokens_{b}.h5"
        with h5py.File(token_file, 'w') as f:
            f.create_dataset('tokens', data=batch_tokens.numpy(), compression=compression)

        metadata['batch_files'].append(str(batch_file))
        metadata['token_files'].append(str(token_file))

        # Force garbage collection after each batch
        del batch_result
        torch.cuda.empty_cache()

    # Save metadata
    metadata_file = save_dir / f"{save_prefix}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata_file




def load_window_data(metadata_file: Union[str, Path], 
                     chunk_indices: Optional[List[int]] = None,
                     print_shapes=False) -> dict:
    """
    Helper function to load the chunked data back into memory when needed.
    Returns a dictionary with the combined data, with all tensors concatenated across chunks.
    
    Args:
        metadata_file: Path to the metadata JSON file
        chunk_indices: Optional list of indices specifying which chunks to load.
                      If None, loads all chunks.
    
    Returns:
        dict: Dictionary containing the concatenated data from specified chunks
    """
    metadata_file = Path(metadata_file)
    base_dir = metadata_file.parent
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Adjust paths to be relative to metadata file location
    metadata['mapping_file'] = str(base_dir / Path(metadata['mapping_file']).name)
    metadata['batch_files'] = [str(base_dir / Path(f).name) for f in metadata['batch_files']]
    metadata['token_files'] = [str(base_dir / Path(f).name) for f in metadata['token_files']]
        
    # Load mapping information
    with h5py.File(metadata['mapping_file'], 'r') as f:
        window_info = {
            'window_starts': torch.from_numpy(f['window_starts'][:]),
            'window_ends': torch.from_numpy(f['window_ends'][:]),
            'attrs': dict(f.attrs)
        }
        
    # Get the chunk indices to load
    total_chunks = len(metadata['batch_files'])
    if chunk_indices is None:
        chunk_indices = list(range(total_chunks))
    else:
        # Validate indices
        if not all(0 <= idx < total_chunks for idx in chunk_indices):
            raise ValueError(f"Chunk indices must be between 0 and {total_chunks-1}")
            
    # Initialize lists to store batch data
    all_tokens = []
    all_window_averages = []
    all_window_p_active = []
    all_feature_acts = []
    all_generated_tokens = []
    
    # Load only specified batches
    for idx in chunk_indices:
        batch_file = metadata['batch_files'][idx]
        token_file = metadata['token_files'][idx]
        
        # Load tokens
        with h5py.File(token_file, 'r') as f:
            all_tokens.append(torch.from_numpy(f['tokens'][:]))
            
        # Load batch data
        with h5py.File(batch_file, 'r') as f:
            # Load window averages
            avg_grp = f['window_averages']
            indices = torch.from_numpy(avg_grp['indices'][:])
            values = torch.from_numpy(avg_grp['values'][:])
            shape = tuple(avg_grp.attrs['shape'])
            avg = torch.sparse_coo_tensor(indices, values, shape)
            all_window_averages.append(avg)
            
            # Load activation probabilities
            p_act_grp = f['window_p_active']
            indices = torch.from_numpy(p_act_grp['indices'][:])
            values = torch.from_numpy(p_act_grp['values'][:])
            shape = tuple(p_act_grp.attrs['shape'])
            p_act = torch.sparse_coo_tensor(indices, values, shape)
            all_window_p_active.append(p_act)
            
            # Load feature activations if present
            if 'feature_acts' in f:
                feat_grp = f['feature_acts']
                indices = torch.from_numpy(feat_grp['indices'][:])
                values = torch.from_numpy(feat_grp['values'][:])
                shape = tuple(feat_grp.attrs['shape'])
                feat_acts = torch.sparse_coo_tensor(indices, values, shape)
                all_feature_acts.append(feat_acts)
                
            # Load generated tokens if present
            if 'tokens_plus_generation' in f:
                all_generated_tokens.append(
                    torch.from_numpy(f['tokens_plus_generation'][:])
                )
    
    # Helper function to safely concatenate sparse tensors
    def concatenate_sparse_tensors(tensor_list):
        if not tensor_list:
            return None
        # Convert sparse tensors to dense before concatenation
        # Note: This might be memory intensive for very large tensors
        return torch.cat([t.to_dense() for t in tensor_list], dim=0)

    tokens = torch.cat(all_tokens, dim=0) if all_tokens else None
    window_averages = concatenate_sparse_tensors(all_window_averages)
    window_p_active = concatenate_sparse_tensors(all_window_p_active)
    feature_acts = concatenate_sparse_tensors(all_feature_acts)
    generated_tokens = torch.cat(all_generated_tokens, dim=0) if all_generated_tokens else None

    if print_shapes:
        print('tokens shape:', tuple(tokens.shape))
        print('window_p_active shape:', tuple(window_p_active.shape))
        print('window_averages shape:', tuple(window_averages.shape))
        if tokens is not None:
            print('tokens shape:', tuple(tokens.shape))
        else:
            print('tokens is None (tokens were not saved)')
        if feature_acts is not None:
            print('feature_acts shape:', tuple(feature_acts.shape))
        else:
            print('feature_acts is None (per-token feature activations were not saved)')
        if generated_tokens is not None:
            print('generated_tokens shape:', tuple(generated_tokens.shape))
        else:
            print('generated_tokens is None (no tokens were generated by the model)')
        print('window_info keys: ', list(window_info.keys()))

    return {
        'window_info': window_info,
        'tokens': all_tokens,
        'window_averages': window_averages,
        'window_p_active': window_p_active,
        'feature_acts': feature_acts,
        'generated_tokens': generated_tokens
    }




import os
import json
import tarfile
import shutil
from pathlib import Path
from typing import Union
from tqdm import tqdm
from huggingface_hub import HfApi, upload_file, hf_hub_download

def safe_cleanup(path: Path):
    """Safely remove a file or directory and its contents."""
    path = Path(path)
    if path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        for item in path.iterdir():
            if item.is_file():
                item.unlink(missing_ok=True)
            elif item.is_dir():
                safe_cleanup(item)
        path.rmdir()

def upload_chunks_to_hub(
    metadata_file: Union[str, Path],
    repo_id: str,
    file_prefix: str = "",
    chunk_size_mb: int = 500,
    cleanup: bool = True,
    token: str = None,
    repo_type: str = "dataset"
) -> dict:
    """
    Uploads chunked window data to Hugging Face Hub with compression.

    Args:
        metadata_file: Path to the metadata JSON file
        repo_id: Hugging Face repository ID (e.g., 'username/repo-name')
        file_prefix: Prefix to add to all uploaded files
        chunk_size_mb: Target size in MB for each compressed archive
        cleanup: Whether to remove local compressed files after upload
        token: Hugging Face API token
        repo_type: Repository type ("dataset" or "model")

    Returns:
        dict: Updated metadata with Hugging Face paths
    """
    try:
        # Initialize Hugging Face API
        api = HfApi(token=token)
        
        # Ensure the repository exists
        try:
            api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
        except Exception as e:
            print(f"Note: Repository already exists or couldn't be created: {e}")

        # Load metadata
        metadata_file = Path(metadata_file)
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Create a temporary directory for compressed files
        temp_dir = Path('./temp_compressed')
        temp_dir.mkdir(exist_ok=True)

        # Helper function to create compressed archives
        def create_tar_archive(files, archive_path):
            with tarfile.open(archive_path, 'w:gz') as tar:
                for file in files:
                    tar.add(file, arcname=os.path.basename(file))

        # Helper function to estimate file size in MB
        def get_size_mb(file_path):
            return os.path.getsize(file_path) / (1024 * 1024)

        # Helper function to create hub path with prefix
        def get_hub_path(filename):
            if file_prefix:
                return f"{file_prefix}/{filename}"
            return filename

        # Group files into chunks based on size
        def group_files_by_size(file_list, target_size_mb):
            groups = []
            current_group = []
            current_size = 0

            for file in file_list:
                file_size = get_size_mb(file)
                if current_size + file_size > target_size_mb and current_group:
                    groups.append(current_group)
                    current_group = [file]
                    current_size = file_size
                else:
                    current_group.append(file)
                    current_size += file_size

            if current_group:
                groups.append(current_group)

            return groups

        # Process and upload files
        hub_paths = {
            'mapping_file': '',
            'batch_archives': [],
            'token_archives': []
        }

        # Upload mapping file
        print("Uploading mapping file...")
        mapping_archive = temp_dir / 'mapping.tar.gz'
        create_tar_archive([metadata['mapping_file']], mapping_archive)
        hub_mapping_path = get_hub_path('mapping.tar.gz')
        api.upload_file(
            path_or_fileobj=str(mapping_archive),
            path_in_repo=hub_mapping_path,
            repo_id=repo_id,
            repo_type=repo_type
        )
        hub_paths['mapping_file'] = hub_mapping_path

        if cleanup:
            safe_cleanup(mapping_archive)

        # Group batch files
        batch_groups = group_files_by_size(metadata['batch_files'], chunk_size_mb)
        token_groups = group_files_by_size(metadata['token_files'], chunk_size_mb)

        # Upload batch files
        print("Uploading batch files...")
        for i, batch_group in enumerate(tqdm(batch_groups)):
            archive_path = temp_dir / f'batch_group_{i}.tar.gz'
            create_tar_archive(batch_group, archive_path)
            hub_archive_path = get_hub_path(f'batch_group_{i}.tar.gz')
            api.upload_file(
                path_or_fileobj=str(archive_path),
                path_in_repo=hub_archive_path,
                repo_id=repo_id,
                repo_type=repo_type
            )
            hub_paths['batch_archives'].append(hub_archive_path)

            if cleanup:
                safe_cleanup(archive_path)

        # Upload token files
        print("Uploading token files...")
        for i, token_group in enumerate(tqdm(token_groups)):
            archive_path = temp_dir / f'token_group_{i}.tar.gz'
            create_tar_archive(token_group, archive_path)
            hub_archive_path = get_hub_path(f'token_group_{i}.tar.gz')
            api.upload_file(
                path_or_fileobj=str(archive_path),
                path_in_repo=hub_archive_path,
                repo_id=repo_id,
                repo_type=repo_type
            )
            hub_paths['token_archives'].append(hub_archive_path)

            if cleanup:
                safe_cleanup(archive_path)

        # Create new metadata with hub paths
        hub_metadata = {
            'original_metadata': metadata,
            'hub_paths': hub_paths,
            'repo_id': repo_id,
            'repo_type': repo_type,
            'file_prefix': file_prefix
        }

        # Save and upload hub metadata
        hub_metadata_path = temp_dir / 'hub_metadata.json'
        with open(hub_metadata_path, 'w') as f:
            json.dump(hub_metadata, f, indent=2)
        
        api.upload_file(
            path_or_fileobj=str(hub_metadata_path),
            path_in_repo=get_hub_path('hub_metadata.json'),
            repo_id=repo_id,
            repo_type=repo_type
        )

        if cleanup:
            safe_cleanup(hub_metadata_path)

        return hub_metadata

    finally:
        # Always attempt cleanup of temp directory at the end
        if cleanup:
            try:
                safe_cleanup(temp_dir)
            except Exception as e:
                print(f"Warning: Could not completely clean up temporary directory: {e}")
                print("You may need to manually remove ./temp_compressed")

def download_chunks_from_hub(
    repo_id: str,
    local_dir: Union[str, Path],
    file_prefix: str = "",
    cleanup: bool = True,
    token: str = None,
    repo_type: str = "dataset"
) -> dict:
    """
    Downloads and extracts chunked window data from Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID (e.g., 'username/repo-name')
        local_dir: Local directory to extract files to
        file_prefix: Prefix used when files were uploaded
        cleanup: Whether to remove compressed archives after extraction
        token: Hugging Face API token
        repo_type: Repository type ("dataset" or "model")

    Returns:
        dict: Original metadata with local file paths
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Create temp directory for archives
    temp_dir = local_dir / 'temp_archives'
    temp_dir.mkdir(exist_ok=True)

    # Helper function to get hub path with prefix
    def get_hub_path(filename):
        if file_prefix:
            return f"{file_prefix}/{filename}"
        return filename

    try:
        # Download and load hub metadata
        hub_metadata_path = hf_hub_download(
            repo_id=repo_id,
            filename=get_hub_path('hub_metadata.json'),
            repo_type=repo_type,
            token=token
        )
        with open(hub_metadata_path, 'r') as f:
            hub_metadata = json.load(f)

        # Initialize lists to track extracted files
        extracted_files = {
            'mapping_file': '',
            'batch_files': [],
            'token_files': []
        }

        # Download and extract mapping file
        print("Downloading mapping file...")
        mapping_archive = temp_dir / 'mapping.tar.gz'
        downloaded_mapping = hf_hub_download(
            repo_id=repo_id,
            filename=hub_metadata['hub_paths']['mapping_file'],
            repo_type=repo_type,
            token=token
        )
        shutil.copy(downloaded_mapping, mapping_archive)
        with tarfile.open(mapping_archive, 'r:gz') as tar:
            tar.extractall(local_dir)
            for member in tar.getmembers():
                if member.isfile():
                    extracted_files['mapping_file'] = str(local_dir / member.name)

        if cleanup:
            safe_cleanup(mapping_archive)

        # Download and extract batch files
        print("Downloading batch files...")
        for archive_path in tqdm(hub_metadata['hub_paths']['batch_archives']):
            local_archive = temp_dir / os.path.basename(archive_path)
            downloaded_archive = hf_hub_download(
                repo_id=repo_id,
                filename=archive_path,
                repo_type=repo_type,
                token=token
            )
            shutil.copy(downloaded_archive, local_archive)
            with tarfile.open(local_archive, 'r:gz') as tar:
                tar.extractall(local_dir)
                for member in tar.getmembers():
                    if member.isfile():
                        extracted_files['batch_files'].append(str(local_dir / member.name))

            if cleanup:
                safe_cleanup(local_archive)

        # Download and extract token files
        print("Downloading token files...")
        for archive_path in tqdm(hub_metadata['hub_paths']['token_archives']):
            local_archive = temp_dir / os.path.basename(archive_path)
            downloaded_archive = hf_hub_download(
                repo_id=repo_id,
                filename=archive_path,
                repo_type=repo_type,
                token=token
            )
            shutil.copy(downloaded_archive, local_archive)
            with tarfile.open(local_archive, 'r:gz') as tar:
                tar.extractall(local_dir)
                for member in tar.getmembers():
                    if member.isfile():
                        extracted_files['token_files'].append(str(local_dir / member.name))

            if cleanup:
                safe_cleanup(local_archive)

        # Sort the files to maintain correct order
        extracted_files['batch_files'].sort()
        extracted_files['token_files'].sort()

        # Create and save local metadata file
        local_metadata_path = local_dir / 'local_metadata.json'
        with open(local_metadata_path, 'w') as f:
            json.dump(extracted_files, f, indent=2)

        print(f"Created local metadata file at: {local_metadata_path}")
        return str(local_metadata_path)

    finally:
        # Always attempt cleanup of temp directory at the end
        if cleanup:
            try:
                safe_cleanup(temp_dir)
            except Exception as e:
                print(f"Warning: Could not completely clean up temporary directory: {e}")