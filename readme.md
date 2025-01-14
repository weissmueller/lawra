# LAIwyer - Rechtsfall Feedback Generator

LAIwyer is a web application that provides AI-generated feedback on legal case descriptions and student inputs. The application uses predefined prompts to analyze the input text and generate feedback based on various criteria such as legal correctness, formal language, and the type of legal statement.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Testing](#testing)
- [License](#license)
- [Authors and Acknowledgments](#authors-and-acknowledgments)
- [Contact Information](#contact-information)

## Features
- Analyze legal case descriptions and student inputs.
- Generate feedback based on predefined prompts.
- Display feedback in a user-friendly format.
- Supports multiple AI providers (OpenAI, Ollama).
- Includes conversation history for context-aware responses.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/username/project.git
    ```
2. Navigate to the project directory:
    ```bash
    cd project
    ```
3. Create a virtual environment:
    ```bash
    python -m venv .venv
    ```
4. Activate the virtual environment:
    - On Windows:
        ```bash
        .venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```
5. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Start the FastAPI server for LocalRag:
    ```bash
    uvicorn localragAPI:app --reload
    ```
2. Start the Flask application:
    ```bash
    python appAPI.py
    ```
3. Open your web browser and navigate to `http://127.0.0.1:5000`.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.