from transformers import T5Config, T5ForConditionalGeneration

def get_model_config(
    vocab_size=32128,
    d_model=512,
    d_kv=64,
    d_ff=2048,
    num_layers=6,
    num_heads=8,
    dropout_rate=0.1,
    pad_token_id=0,
    eos_token_id=1,
    decoder_start_token_id=None,
):
    """
    Returns a T5Config for the model. 
    Defaults are close to T5-Small but can be adjusted.
    """
    if decoder_start_token_id is None:
        decoder_start_token_id = pad_token_id
    config = T5Config(
        vocab_size=vocab_size,
        d_model=d_model,
        d_kv=d_kv,
        d_ff=d_ff,
        num_layers=num_layers,  # Encoder layers
        num_decoder_layers=num_layers, # Decoder layers
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        decoder_start_token_id=decoder_start_token_id,  # Typically pad or specifically start
    )
    return config

def get_model(
    config=None,
    vocab_size=32128,
    pad_token_id=0,
    eos_token_id=1,
    decoder_start_token_id=None,
):
    """
    Initializes a T5ForConditionalGeneration model from scratch.
    """
    if config is None:
        config = get_model_config(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
        )
        
    model = T5ForConditionalGeneration(config)
    return model

if __name__ == "__main__":
    # Test initialization
    print("Initializing model...")
    model = get_model()
    print(f"Model config: {model.config}")
    print(f"Parameters: {model.num_parameters()}")
    print("Model ready.")
