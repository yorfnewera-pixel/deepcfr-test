def _resolve_model_save_path(path_prefix, iteration):
    if str(path_prefix).endswith(".pt"):
        return str(path_prefix)
    return f"{path_prefix}_iteration_{iteration}.pt"


def _strip_legacy_bucket_head_keys(state_dict):
    if state_dict is None:
        return None
    return {k: v for k, v in state_dict.items() if not k.startswith('bucket_head.')}
