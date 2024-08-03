def reverse_argsort(indices):
    n = len(indices)
    reverse_indices = [0] * n
    for i, idx in enumerate(indices):
        reverse_indices[idx] = i
    return reverse_indices


MAX_N_OUTLIERS_TEXT = 20
train_only_message = "This function is only available for training data."
