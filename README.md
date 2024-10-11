# RAG-for-BUSINESS-DOCS

This project is a Python-based application that uses LangChain, Chroma, and GPT4All embeddings to process PDF files, split them into chunks, and store them in a vector database. It can be run locally or using Docker.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Local Setup](#local-setup)
  - [Docker Setup](#docker-setup)
- [Usage](#usage)
  - [Running Locally](#running-locally)
  - [Running with Docker](#running-with-docker)
- [Environment Variables](#environment-variables)
- [Features](#features)
- [Gitignore Setup](#gitignore-setup)
- [Contributing](#contributing)
- [License](#license)

---

## Prerequisites

Before running the application, make sure you have the following installed:

### For Local Setup:
- **Python 3.11.6**
- **pip** (Python package manager)

### For Docker Setup:
- **Docker** (Ensure Docker is installed and running)

---

## Installation

### Local Setup

1. **Clone the repository**:

    ```bash
    git clone https://github.com/your_username/your_project.git
    cd your_project
    ```

2. **Set up a virtual environment** (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    - Create a `.env` file in your project directory with the necessary environment variables.
    - Follow the [Environment Variables](#environment-variables) section for guidance.

### Docker Setup

1. **Clone the repository**:

    ```bash
    git clone https://github.com/your_username/your_project.git
    cd your_project
    ```

2. **Build the Docker image**:

    ```bash
    docker build -t my_project_image .
    ```

3. **Run the Docker container**:

    ```bash
    docker run -p 8501:8501 --env-file /path/to/.env my_project_image
    ```

---

## Usage

### Running Locally

1. **Run the application**:

    Depending on whether your app uses Streamlit or plain Python, use one of the following commands:

    ```bash
    streamlit run app.py
    ```

    or

    ```bash
    python app.py
    ```

2. Open your browser and navigate to `http://localhost:8501` to access the application (for Streamlit). If you're using another framework, change the port accordingly.

### Running with Docker

1. **Start the Docker container**:

    ```bash
    docker run -p 8501:8501 --env-file /path/to/.env my_project_image
    ```

2. Open your browser and navigate to `http://localhost:8501` to access the application.

---

## Environment Variables

To configure the environment, create a `.env` file in your project directory and add your key-value pairs like this:

```bash
# .env file
API_KEY=your_api_key
```

- This `.env` file will not be committed to the Git repository, keeping your sensitive information safe.

---

## Features

- **PDF Processing**: Upload a PDF file and split it into manageable text chunks.
- **Vector Storage**: Store the processed chunks in a vector database using Chroma and GPT4All embeddings.
- **Streamlit UI**: (Optional) Use a web interface to upload PDFs and visualize the chunked output.

---

### Stop Tracking the `.env` File:

If you’ve already committed the `.env` file before setting up the `.gitignore`, don’t worry! Here’s how to fix it:

```bash
git rm --cached .env
git commit -m "Remove .env from tracking and add it to .gitignore"
```

---

## Contributing

Feel free to fork this repository, make some changes, and submit pull requests. Every contribution helps make this project even better!