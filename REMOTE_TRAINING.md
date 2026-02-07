# Remote Training with Modal

This project supports offloading training to cloud GPUs using [Modal](https://modal.com/).

## Prerequisites

1.  **Create a Modal Account**: Sign up at [modal.com](https://modal.com/).
2.  **Install Modal Client**:
    ```bash
    pip install modal
    ```
3.  **Authenticate**:
    ```bash
    modal setup
    ```

## Running Training Remotely

To run the training loop on a cloud GPU (configured to use "any" GPU, typically T4 or A10G), run:

```bash
modal run src/training/modal_train.py
```

### Customization

You can pass arguments to the entrypoint:

```bash
modal run src/training/modal_train.py --epochs 5
```

### Configuration Details

-   **Code & Data**: The script mounts your local `src`, `data`, and `models` directories to the remote container at `/root/`.
-   **GPU**: Configured to use any available GPU. You can change `gpu="any"` to `gpu="A10G"` in `src/training/modal_train.py` for more power.
-   **Dependencies**: Defined in the `image` variable in `modal_train.py` (installs torch, transformers, etc.).

## Persistence Note

Currently, the script mounts local folders for input. Outputs written to `/root/models` inside the container **will be lost** when the container shuts down unless you:
1.  Use `modal.Volume` to persist data.
2.  Or modify the script to return the model file bytes and save them locally.
3.  Or verify functionality first (current setup) and then enable Volume storage.
