# Use an official Python runtime as a base image
FROM python:3.12-slim

# Optional: create a non-root user (recommended for production)
# RUN useradd -ms /bin/bash appuser
# USER appuser

# Install your library from PyPI
RUN pip install fluke-fl

# Default command (optional)
CMD ["python3"]
