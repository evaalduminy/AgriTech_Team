import os
import shutil

ARCHIVE_DIR = "_archive_2026-10-25"
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# 1. Permanent deletions (Cache and empty test files)
del_targets = [
    "rust_test.jpg",
    "__pycache__",
    ".pytest_cache",
    os.path.join("scripts", "__pycache__")
]
for t in del_targets:
    if os.path.exists(t):
        if os.path.isdir(t):
            shutil.rmtree(t, ignore_errors=True)
        else:
            os.remove(t)
        print(f"Deleted cache/dummy: {t}")

# 2. Files to preserve in root
KEEP_FILES = {
    "train_final_fixed.py",
    "train_colab.py",
    "outputs",
    "scripts",
    "labels.txt",
    "requirements.txt",
    "requirements_v3.txt",
    "pyproject.toml",
    ".gitignore",
    ".github",
    "Dockerfile",
    "api_service.py",
    "AgriTech_Colab_Training.ipynb",
    "cnn-all.ipynb",
    "kernel-metadata.json",
    ARCHIVE_DIR,
    "README_Supervisor.md"
}

# 3. Move items to archive
items = [item for item in os.listdir(".") if item not in KEEP_FILES]
moved_count = 0
for item in items:
    src = item
    dst = os.path.join(ARCHIVE_DIR, item)
    try:
        shutil.move(src, dst)
        moved_count += 1
        print(f"Archived: {item} -> {dst}")
    except Exception as e:
        print(f"Error moving {item}: {e}")

print(f"\nSuccessfully archived {moved_count} items to {ARCHIVE_DIR}/")

# 4. Update .gitignore to exclude archive directory
gitignore_path = ".gitignore"
if os.path.exists(gitignore_path):
    content = open(gitignore_path, "r", encoding="utf-8").read()
    if "_archive" not in content:
        with open(gitignore_path, "a", encoding="utf-8") as f:
            f.write("\n\n# Archive directories\n_archive*/\n")
        print("Updated .gitignore with _archive*/")
