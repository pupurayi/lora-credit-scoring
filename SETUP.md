# Setup Guide

Follow these steps once, in order. Total time: ~15 minutes plus dataset download time (~1 GB Home Credit zip).

## 1. Open a terminal in this folder

Windows PowerShell:
```powershell
cd "$HOME\Downloads\ESWA_FullLength_Package\code"
```

## 2. Create the Python environment

You need Python 3.10 or newer. Verify with `python --version`.

Windows:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/Mac:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This will take ~5 minutes (downloads PyTorch, XGBoost, SHAP, etc.).

## 3. Set up your Kaggle API token

This is the step you don't have yet.

1. Go to <https://www.kaggle.com/settings/account> (must be logged in).
2. Scroll to the "API" section.
3. Click **Create New Token**. A file `kaggle.json` downloads.
4. Place it here:
   - Windows: `%USERPROFILE%\.kaggle\kaggle.json`
     (you may need to create the `.kaggle` folder first)
   - Linux/Mac: `~/.kaggle/kaggle.json` and run `chmod 600 ~/.kaggle/kaggle.json`
5. Accept the competition rules at <https://www.kaggle.com/c/home-credit-default-risk/rules>
   (click "I Understand and Accept"). You can't download until you do.

Verify it works:
```
kaggle competitions list
```
You should see a list of recent competitions. If you get "403 Forbidden" it means the token isn't installed correctly.

## 4. Download both datasets

From the `code/` folder with the venv activated:

```
python data/download_german_credit.py
python data/download_home_credit.py
```

German Credit is ~200 KB, downloads in seconds.
Home Credit is ~700 MB zipped, ~2.7 GB extracted. Expect 2-10 minutes depending on your connection.

Both scripts will verify row counts after download and abort if anything's off.

## 5. Ping me

Once both scripts say `[ok] ... ready in ...`, come back and tell me. I'll write the preprocessing scripts next.

## Troubleshooting

**`pip install` fails on `torch`**: PyTorch is large. If install fails, try:
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
This explicitly uses the CPU-only wheel, which is what we want.

**`pip install` fails on `xgboost` on Windows**: Install Microsoft Visual C++ Redistributable first: <https://aka.ms/vs/17/release/vc_redist.x64.exe>

**Kaggle API says `403`**: You haven't accepted the competition rules (Step 3.5).

**Out of disk space**: Home Credit extracts to ~2.7 GB. Make sure your drive has at least 5 GB free.
