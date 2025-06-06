from huggingface_hub import upload_file

repo_id = "Mehardeep7/transformer-attention-model"
model_path = "opus_books_weights/tmodel_30.pt"

upload_file(
    path_or_fileobj=model_path,
    path_in_repo="tmodel_30.pt",
    repo_id=repo_id,
    repo_type="model"
)

