#@title Code to make dashboards
def get_top_windows_unique_contexts(window_averages: torch.Tensor, feature_id: int, top_k: int = 8):
    """
    Find windows where a specific feature has highest activation, taking only one window per batch.

    Args:
        window_averages: Sparse COO tensor of shape [batch_size, num_windows, num_features]
        feature_id: The feature to analyze
        top_k: Number of top windows to return
    """
    # Convert sparse to dense for the specific feature we're interested in
    # First, find indices that match our feature_id
    feature_mask = window_averages._indices()[2] == feature_id
    relevant_indices = window_averages._indices()[:, feature_mask]
    relevant_values = window_averages._values()[feature_mask]

    # Create list of (batch_idx, window_idx, activation) tuples
    windows = [(relevant_indices[0, i].item(),
               relevant_indices[1, i].item(),
               relevant_values[i].item())
              for i in range(relevant_values.size(0))]

    # Sort by activation value
    windows.sort(key=lambda x: x[2], reverse=True)

    # Take top_k while ensuring one window per batch
    seen_batches = set()
    top_windows = []

    for window in windows:
        if window[0] not in seen_batches and len(top_windows) < top_k:
            top_windows.append(window)
            seen_batches.add(window[0])

    return top_windows

def print_top_windows_html(tokens, feature_acts, window_averages, feature_id,
                         window_length: int = 64, stride: int = 16,
                         num_windows_to_show: int = 6, context_tokens: int = 0,
                         min_opacity: float = 0.2):
    """
    Generate HTML visualization with translucent highlighting based on sparse activation tensors.

    Args:
        tokens: Token ids [batch_size, seq_len]
        feature_acts: Sparse COO tensor of shape [batch_size, seq_len, num_features]
        window_averages: Sparse COO tensor of shape [batch_size, num_windows, num_features]
        feature_id: Feature to visualize
        window_length: Length of each window
        stride: Stride between windows
        num_windows_to_show: Number of top windows to display
        context_tokens: Number of tokens to show before and after window
        min_opacity: Minimum opacity for highlighting
    """
    # Get device from input tensors
    device = feature_acts.device

    # Get top windows for this feature
    windows = get_top_windows_unique_contexts(window_averages, feature_id, num_windows_to_show)

    # Convert sparse feature_acts to dense for the specific feature we're interested in
    feature_mask = feature_acts._indices()[2] == feature_id
    feature_indices = feature_acts._indices()[:, feature_mask]
    feature_values = feature_acts._values()[feature_mask]

    html_output = f"""
    <div class="windows-container" style="font-family: monospace;">
        <h2>Feature {feature_id} Top Windows</h2>
        <style>
            .token {{
                display: inline;
                padding: 0;
                margin: 0;
                position: relative;
            }}
            .token:hover .tooltip {{
                display: block;
            }}
            .tooltip {{
                display: none;
                position: absolute;
                background: #333;
                color: white;
                padding: 0px 0px;
                border-radius: 0px;
                font-size: 12px;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                white-space: nowrap;
                z-index: 1;
            }}
            .window-box {{
                margin: 20px 0;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background: white;
            }}
            .text-container {{
                font-size: 0;
                word-spacing: 0;
                letter-spacing: 0;
            }}
            .stats {{
                color: #666;
                font-size: 0.9em;
                margin-top: 8px;
            }}
        </style>
    """

    def get_color(act_value, max_abs_act):
        """Generate translucent color based on activation value."""
        if act_value == 0:
            return "rgba(0, 0, 0, 0)"

        opacity = min(1.0, max(min_opacity, abs(act_value) / max_abs_act))

        if act_value > 0:
            return f"rgba(255, 0, 0, {opacity})"
        else:
            return f"rgba(0, 0, 255, {opacity})"

    for batch_idx, window_idx, window_activation in windows:
        start_idx = window_idx * stride
        end_idx = start_idx + window_length

        context_start = max(0, start_idx - context_tokens)
        context_end = min(tokens.shape[1], end_idx + context_tokens)

        window_tokens = tokens[batch_idx, context_start:context_end].tolist()

        # Get activations for this batch and token range
        batch_mask = feature_indices[0] == batch_idx
        token_mask = (feature_indices[1] >= context_start) & (feature_indices[1] < context_end)
        window_mask = batch_mask & token_mask

        # Create activation array, initially all zeros
        token_activations = torch.zeros(context_end - context_start, device=device, dtype=feature_acts.dtype)

        # Fill in non-zero activations
        rel_positions = feature_indices[1, window_mask] - context_start
        token_activations[rel_positions] = feature_values[window_mask]

        # Move to CPU for visualization
        token_activations = token_activations.cpu()

        # Convert to list for easier processing
        token_activations = token_activations.tolist()

        # Calculate maximum absolute activation for scaling
        window_acts = token_activations[max(0, start_idx-context_start):min(end_idx-context_start, len(token_activations))]
        max_abs_act = max(abs(act) for act in window_acts if act != 0) if any(act != 0 for act in window_acts) else 1

        html_output += f"""
        <div class="window-box">
            <div>Batch {batch_idx}, Window {start_idx}-{end_idx}</div>
            <div>Window activation: {window_activation:.4f}</div>
            <div style="font-size: 0.8em">Max activation magnitude: {max_abs_act:.4f}</div>
            <div class="text-container">
        """

        for pos, (token, act_value) in enumerate(zip(window_tokens, token_activations)):
            text = model.tokenizer.decode([token])

            abs_pos = context_start + pos
            is_in_window = start_idx <= abs_pos < end_idx

            bg_color = get_color(act_value, max_abs_act) if is_in_window else "rgba(0, 0, 0, 0)"

            tooltip_text = f"Activation: {act_value:.4f}"

            html_output += f"""
                <span class="token" style="background-color: {bg_color}; color: black; font-size: 16px;">
                    {html.escape(text)}
                    <span class="tooltip">{tooltip_text}</span>
                </span>"""

        html_output += f"""
            </div>
            <div class="stats">
            </div>
        </div>
        """

    print(f"Activation density: { 100 * feature_values.shape[0] / (feature_acts.shape[0] * feature_acts.shape[1]) :.3f }%")

    html_output += "</div>"
    return HTML(html_output)


    

def print_top_logits(sae, feature_id, num_to_show=8):
    dec = sae.decoder.weight[:, feature_id]
    logits = dec @ model.W_U  # shape [n_vocab]
    top_logits, top_indices = torch.topk(logits, num_to_show)
    print(f"Top logits for feature {feature_id}:")
    for i in range(num_to_show):
        print(f"{top_logits[i].item():.2f}   {model.tokenizer.decode(top_indices[i])}")