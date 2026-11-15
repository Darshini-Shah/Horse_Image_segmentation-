import json
import os

with open('code_payload.b64') as f:
    b64_payload = f.read().strip()

notebook = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# Horse Segmentation — Skip Connections in Encoder-Decoder\n',
                '**Self-contained notebook for Google Colab & VS Code Colab Extension**\n',
                '\n',
                '> **Runtime setup:** In the top-right kernel selector, ensure you are connected to **Google Colab (T4 GPU)**.\n',
                '> **Zero manual uploads needed!** Cell 1 automatically restores the full project codebase (`src/`, `configs/`, `main.py`) directly onto the Colab machine.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# 1. Verify GPU availability\n',
                '!nvidia-smi --query-gpu=name,memory.total --format=csv'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 1. Auto-Restore Project Source Code & Install Dependencies\n',
                'Extracts all necessary modules (`src/`, `configs/`, `main.py`, `requirements.txt`) into `/content` in less than 1 second, so you do not need to upload anything.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'import os, sys, io, base64, zipfile\n',
                '\n',
                '# Auto-restore source code onto Colab\n',
                'if not os.path.exists("src/models.py"):\n',
                '    print("--> Restoring project source code onto Colab filesystem...")\n',
                f'    CODE_B64 = "{b64_payload}"\n',
                '    with zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_B64))) as z:\n',
                '        z.extractall(".")\n',
                '    print("✔ All source code files restored successfully!")\n',
                'else:\n',
                '    print("✔ Project source code is already present.")\n',
                '\n',
                '# Ensure current working directory is on sys.path so "src" is always found\n',
                'cwd = os.path.abspath(".")\n',
                'if cwd not in sys.path:\n',
                '    sys.path.insert(0, cwd)\n',
                '\n',
                '# Install dataset helper and YAML parser\n',
                '!pip -q install kagglehub pyyaml\n',
                'print("✔ Environment ready.")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 2. Download Weizmann Horse Dataset (327 Pairs)\n',
                'Downloads the dataset using `kagglehub` directly onto Colab in seconds.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'import os, shutil, kagglehub\n',
                '\n',
                '# Check if dataset is already present or needs downloading\n',
                'if not os.path.exists("data/cache_256.npz") and not os.path.exists("data/raw") and not os.path.exists("data/horse"):\n',
                '    print("--> Downloading Weizmann Horse dataset via kagglehub...")\n',
                '    path = kagglehub.dataset_download("ztaihong/weizmann-horse-database")\n',
                '    print("✔ Downloaded to:", path)\n',
                '    os.makedirs("data", exist_ok=True)\n',
                '    if not os.path.exists("data/raw"):\n',
                '        try:\n',
                '            os.symlink(path, "data/raw")\n',
                '        except Exception:\n',
                '            shutil.copytree(path, "data/raw", dirs_exist_ok=True)\n',
                '\n',
                '# Verify image-mask pairs\n',
                'from src.data import discover_pairs\n',
                'pairs, unpaired = discover_pairs("data/raw") if os.path.exists("data/raw") else discover_pairs("data")\n',
                'print(f"✔ Successfully discovered {len(pairs)} matched image/mask pairs!")\n',
                'if pairs:\n',
                '    print("Example pair:", pairs[0])'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 3. Run Pipeline on T4 GPU\n',
                '- **Option A (Quick Sanity Run)**: ~15-20 min on T4 GPU (1 seed, 40 epochs, tuning).\n',
                '- **Option B (Full Experiment)**: ~2-3 h on T4 GPU (3 seeds, 120 epochs, full tuning).'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# Option A: Quick Sanity Run (~15-20 min on T4 GPU)\n',
                '!python main.py --stage all --config configs/quick.yaml --results_dir results_quick'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# Option B: Full Experiment (~2-3 h on T4 GPU for final paper metrics)\n',
                '# Uncomment the line below to run the full experiment:\n',
                '# !python main.py --stage all --config configs/default.yaml --results_dir results'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 4. Visualize Results & Key Figures\n',
                'Renders the executive summary table and all evaluation figures.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'from IPython.display import Markdown, Image, display\n',
                'import os\n',
                '\n',
                '# Choose active results directory\n',
                'RESULTS_DIR = "results_quick" if os.path.exists("results_quick/summary.md") else "results"\n',
                'summary_path = f"{RESULTS_DIR}/summary.md"\n',
                '\n',
                'if os.path.exists(summary_path):\n',
                '    display(Markdown(open(summary_path).read()))\n',
                'else:\n',
                '    print(f"{summary_path} not found. Please run the training stage first.")'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# Display comparison figures\n',
                'FIGS = [\n',
                '    "architecture", "results_main_metrics", "outline_two_examples",\n',
                '    "error_near_outline", "skip_knockout", "training_curves",\n',
                '    "results_by_horse_size", "qualitative_random", "qualitative_failures", "efficiency"\n',
                ']\n',
                'for f in FIGS:\n',
                '    p = f"{RESULTS_DIR}/figures/{f}.png"\n',
                '    if os.path.exists(p):\n',
                '        title = f.upper().replace("_", " ")\n',
                '        print(f"\\n=== {title} ===")\n',
                '        display(Image(p, width=900))'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 5. Export Results Archive (Download to Local PC or Google Drive)\n',
                'Packages all tables, figures, logs, and summaries for your report submission.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# 1. Package results into horse_results.zip\n',
                '!zip -q -r horse_results.zip {RESULTS_DIR} -x "*/checkpoints/*" "*/predictions/*"\n',
                'print("✔ horse_results.zip generated successfully!")\n',
                '\n',
                '# 2. Try browser download (works in Colab web UI)\n',
                'try:\n',
                '    from google.colab import files\n',
                '    files.download("horse_results.zip")\n',
                'except Exception:\n',
                '    print("Note: In VS Code, open the Colab file tree on the left sidebar, right-click horse_results.zip -> Download.")\n',
                '\n',
                '# 3. Optional: Direct save to your Google Drive\n',
                '# from google.colab import drive\n',
                '# drive.mount("/content/drive")\n',
                '# !cp horse_results.zip /content/drive/MyDrive/\n',
                '# print("✔ horse_results.zip copied directly to your Google Drive!")'
            ]
        }
    ],
    'metadata': {
        'accelerator': 'GPU',
        'colab': {
            'gpuType': 'T4',
            'provenance': []
        },
        'kernelspec': {
            'display_name': 'Python 3',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 0
}

with open('notebooks/Horse_Segmentation_Colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print('Updated notebooks/Horse_Segmentation_Colab.ipynb successfully!')

