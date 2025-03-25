class Config:
    model_path = "gpt2"
    dataset_name = "OpenAssistant/oasst1"
    cache_dir = "./input_data"
    max_length = 2048
    batch_size = 16
    learning_rate = 3e-5
    num_epochs = 10
    pad_token_id = 0
    early_stopping_patience = 2  # Stop training if no improvement for N epochs
