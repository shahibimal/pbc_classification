# Peripheral Blood Cell Classification

## Getting Started

### Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/prabhashj07/pbc_classification.git
    cd pbc_classification
    ```

2. Set up a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Add environment variables:

    ```bash
    cp .env.example .env
    ```

### Dataset Preparation

Dataset: [Google Drive Link](https://drive.google.com/file/d/1LskXwZL6fphjF-VbudPMa6CbPUX_v0ZC)

To prepare the dataset, run the following command:
It will download the dataset from Google Drive and extract it to the `data/` directory.

```bash
make dataset
``````
### Training

Run the training script using one of the following methods:

```bash
python train.py

# switch model and placeholders
python train.py --model_name <model_name> --batch_size <batch_size> --epochs <epochs> --lr <learning_rate> [--use_scheduler]
```

Replace the placeholders with appropriate values:

- `<model_name>`: The model you wish to train (e.g., vit_base_patch16_224, BloodNetViT, etc.).
- `<batch_size>`: The batch size for training (default is 32).
- `<epochs>`: The number of epochs for training (default is 100).
- `<learning_rate>`: The learning rate (default is 2e-5).
- `--use_scheduler`: Use this flag to enable a learning rate scheduler (optional).

Example to train the `BloodNetViT` model:
```bash
python train.py --model_name BloodNetViT --batch_size 64 --epochs 50 --lr 1e-3 --use_scheduler
```

Available models with their keys in the factory are:

#### ViT (Vision Transformer) Models:

- `vit_base_patch16_224`
- `vit_small_patch16_224`
- `vit_base_patch32_224`
- `vit_large_patch16_224`
- `vit_large_patch32_224`
- `vit_tiny_patch16_224`

#### Custom Model:

- `BloodNetViT`

### Project Structure

- `data/`: Contains the dataset used for training and testing.
- `src/`: Contains the source code of the project.
- `scripts/`: Contains utility scripts, such as downloading datasets.
- `artifacts/`: Contains trained model checkpoints.
- `train.py`: Main training script.
- `requirements.txt`: List of required packages.

## License

This project is licensed under the [MIT License](LICENSE).