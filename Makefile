# --- Cross-Platform Setup ---
# On Windows, set the shell to PowerShell to ensure conda commands work correctly.
# On Linux/macOS, this block is ignored, and the default shell is used.
ifeq ($(OS), Windows_NT)
    SHELL := powershell.exe
endif

# --- Makefile Targets ---
venv-init:
	@echo "Creating venv using conda..."
	conda create --prefix ./venv python=3.9 -y

	@echo "Installing cudatoolkit and cudnn..."
	conda run --prefix ./venv conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y

	@echo "Upgrading pip to the latest version..."
	conda run --prefix ./venv pip install --upgrade pip

	@echo "Venv setup complete."