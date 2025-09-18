# 1. Start from a standard Python base image
FROM python:3.13

# 2. Set the root directory for the app inside the container
WORKDIR /app

# 3. Copy the files that define your project and its dependencies
COPY pyproject.toml requirements.txt ./

# 4. Install all the locked dependencies from your requirements file
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your project's source code into the container
COPY src/ /app/src/

# 6. Specify the default command to run your main pipeline
CMD ["python", "src/project_name/pipelines/training_pipeline.py"]