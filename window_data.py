from pathlib import Path
import h5py
import torch
from tqdm import tqdm
import numpy as np
import json
from typing import Optional, Tuple, Union, List, NamedTuple



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
            avg_grp.create_dataset('indices', data=avg.indices().numpy(), compression=compression)
            avg_grp.create_dataset('values', data=avg.values().float().numpy(), compression=compression)
            avg_grp.attrs['shape'] = avg.size()

            # Save activation probabilities
            p_act_grp = f.create_group('window_p_active')
            p_act_grp.create_dataset('indices', data=p_act.indices().numpy(), compression=compression)
            p_act_grp.create_dataset('values', data=p_act.values().float().numpy(), compression=compression)
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





#@title function to move data from disk to CPU
def load_window_data(metadata_file: Union[str, Path]) -> dict:
    """
    Helper function to load the chunked data back into memory when needed.
    Returns a dictionary with the combined data.
    """
    metadata_file = Path(metadata_file)
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Load mapping information
    with h5py.File(metadata['mapping_file'], 'r') as f:
        window_info = {
            'window_starts': torch.from_numpy(f['window_starts'][:]),
            'window_ends': torch.from_numpy(f['window_ends'][:]),
            'attrs': dict(f.attrs)
        }

    # Initialize lists to store batch data
    all_tokens = []
    all_window_averages = []
    all_window_p_active = []
    all_feature_acts = []
    all_generated_tokens = []

    # Load each batch
    for batch_file, token_file in zip(metadata['batch_files'], metadata['token_files']):
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

    return {
        'window_info': window_info,
        'tokens': torch.cat(all_tokens, dim=0),
        'window_averages': all_window_averages,
        'window_p_active': all_window_p_active,
        'feature_acts': all_feature_acts if all_feature_acts else None,
        'generated_tokens': all_generated_tokens if all_generated_tokens else None
    }




#@title Upload to google drive
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

def upload_chunks_to_drive(
    metadata_file: Union[str, Path],
    drive_folder: str = "window_data",
    chunk_size_mb: int = 500,
    cleanup: bool = True
) -> dict:
    """
    Uploads chunked window data to Google Drive with compression.

    Args:
        metadata_file: Path to the metadata JSON file
        drive_folder: Folder name in Google Drive to store the files
        chunk_size_mb: Target size in MB for each compressed archive
        cleanup: Whether to remove local compressed files after upload

    Returns:
        dict: Updated metadata with Google Drive paths
    """
    try:
        # Mount Google Drive
        drive.mount('/content/drive')
        drive_root = Path('/content/drive/MyDrive')

        # Create drive folder if it doesn't exist
        drive_path = drive_root / drive_folder
        drive_path.mkdir(parents=True, exist_ok=True)

        # Load metadata
        metadata_file = Path(metadata_file)
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Create a temporary directory for compressed files
        temp_dir = Path('/content/temp_compressed')
        temp_dir.mkdir(exist_ok=True)

        # Helper function to create compressed archives
        def create_tar_archive(files, archive_path):
            with tarfile.open(archive_path, 'w:gz') as tar:
                for file in files:
                    tar.add(file, arcname=os.path.basename(file))

        # Helper function to estimate file size in MB
        def get_size_mb(file_path):
            return os.path.getsize(file_path) / (1024 * 1024)

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
        drive_paths = {
            'mapping_file': '',
            'batch_archives': [],
            'token_archives': []
        }

        # Upload mapping file
        print("Uploading mapping file...")
        mapping_archive = temp_dir / 'mapping.tar.gz'
        create_tar_archive([metadata['mapping_file']], mapping_archive)
        drive_mapping_path = drive_path / 'mapping.tar.gz'
        shutil.copy(mapping_archive, drive_mapping_path)
        drive_paths['mapping_file'] = str(drive_mapping_path)

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
            drive_archive_path = drive_path / f'batch_group_{i}.tar.gz'
            shutil.copy(archive_path, drive_archive_path)
            drive_paths['batch_archives'].append(str(drive_archive_path))

            if cleanup:
                safe_cleanup(archive_path)

        # Upload token files
        print("Uploading token files...")
        for i, token_group in enumerate(tqdm(token_groups)):
            archive_path = temp_dir / f'token_group_{i}.tar.gz'
            create_tar_archive(token_group, archive_path)
            drive_archive_path = drive_path / f'token_group_{i}.tar.gz'
            shutil.copy(archive_path, drive_archive_path)
            drive_paths['token_archives'].append(str(drive_archive_path))

            if cleanup:
                safe_cleanup(archive_path)

        # Create new metadata with drive paths
        drive_metadata = {
            'original_metadata': metadata,
            'drive_paths': drive_paths
        }

        # Save drive metadata
        drive_metadata_path = drive_path / 'drive_metadata.json'
        with open(drive_metadata_path, 'w') as f:
            json.dump(drive_metadata, f, indent=2)

        return drive_metadata

    finally:
        # Always attempt cleanup of temp directory at the end
        if cleanup:
            try:
                safe_cleanup(temp_dir)
            except Exception as e:
                print(f"Warning: Could not completely clean up temporary directory: {e}")
                print("You may need to manually remove /content/temp_compressed")

                

#@title Download from drive
def download_chunks_from_drive(
    drive_metadata_path: Union[str, Path],
    local_dir: Union[str, Path],
    cleanup: bool = True
) -> dict:
    """
    Downloads and extracts chunked window data from Google Drive.

    Args:
        drive_metadata_path: Path to the drive metadata JSON file
        local_dir: Local directory to extract files to
        cleanup: Whether to remove compressed archives after extraction

    Returns:
        dict: Original metadata with local file paths
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Load drive metadata
    with open(drive_metadata_path, 'r') as f:
        drive_metadata = json.load(f)

    # Create temp directory for archives
    temp_dir = local_dir / 'temp_archives'
    temp_dir.mkdir(exist_ok=True)

    # Initialize lists to track extracted files
    extracted_files = {
        'mapping_file': '',
        'batch_files': [],
        'token_files': []
    }

    try:
        # Download and extract mapping file
        print("Downloading mapping file...")
        mapping_archive = temp_dir / 'mapping.tar.gz'
        shutil.copy(drive_metadata['drive_paths']['mapping_file'], mapping_archive)
        with tarfile.open(mapping_archive, 'r:gz') as tar:
            tar.extractall(local_dir)
            # Get the name of the extracted mapping file
            for member in tar.getmembers():
                if member.isfile():
                    extracted_files['mapping_file'] = str(local_dir / member.name)

        if cleanup:
            safe_cleanup(mapping_archive)

        # Download and extract batch files
        print("Downloading batch files...")
        for archive_path in tqdm(drive_metadata['drive_paths']['batch_archives']):
            local_archive = temp_dir / os.path.basename(archive_path)
            shutil.copy(archive_path, local_archive)
            with tarfile.open(local_archive, 'r:gz') as tar:
                tar.extractall(local_dir)
                # Track extracted batch files
                for member in tar.getmembers():
                    if member.isfile():
                        extracted_files['batch_files'].append(str(local_dir / member.name))

            if cleanup:
                safe_cleanup(local_archive)

        # Download and extract token files
        print("Downloading token files...")
        for archive_path in tqdm(drive_metadata['drive_paths']['token_archives']):
            local_archive = temp_dir / os.path.basename(archive_path)
            shutil.copy(archive_path, local_archive)
            with tarfile.open(local_archive, 'r:gz') as tar:
                tar.extractall(local_dir)
                # Track extracted token files
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