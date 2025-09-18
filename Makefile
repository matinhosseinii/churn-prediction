# Install all Python dependencies from the locked requirements file

install:
	@echo "Installing dependencies from requirements.txt..."
	pip install -r requirements.txt

# Use make install in git bash to run it
# You can add some command shortcuts