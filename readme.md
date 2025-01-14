# LaWrA - AI Writing Assistant for Law Students

Lawra is a web application designed to provide AI-generated feedback to legal students on predefined cases. The application leverages OpenAI's API to analyze student submissions and generate detailed feedback on their legal writing and case analysis.

## Features

- **AI-Generated Feedback**: Provides feedback on legal writing style, citation accuracy, and case analysis.
- **Predefined Cases**: Students can select from a list of predefined cases to receive feedback on.
- **Real-Time Updates**: Feedback is streamed to the client in real-time using Socket.IO.
- **Cache System**: Caching mechanism to store and retrieve feedback responses.

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package installer)
- Node.js and npm (for running the frontend)

### Setup

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/lawra.git
   cd lawra
   ```

2. Install the required Python packages:

   ```sh
   pip install -r requirements.txt
   ```

3. Set up environment variables:

   - Create a `.env` file in the root directory.
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```

4. Install frontend dependencies:

   ```sh
   cd frontend
   npm install
   ```

5. Build the frontend:

   ```sh
   npm run build
   ```

## Usage

1. Before starting, create a `cases.json` file in the root directory. This file should be a copy of `cases_template.json` but populated with actual legal cases.

2. Start the backend server:

   ```sh
   python proofreader.py
   ```

3. Open your web browser and navigate to `http://127.0.0.1:5000`.

## File Structure

- `cases_template.json`: Template for defining cases.
- `cases.json`: JSON file containing the actual legal cases created based on `cases_template.json`.
- `legal_texts.json`: JSON file containing sample legal texts.
- `proofreader.py`: Main backend application file.
- `proofreader_prompts.json`: JSON file containing prompt configurations for generating feedback.
- `requirements.txt`: List of Python dependencies.
- `static/`: Directory containing static assets like icons and images.
  - `icon1.png`
  - `icon2.png`
  - `loading.png`
- `templates/`: Directory containing HTML templates.
  - `index.html`
- `readme.md`: Documentation file.

## Configuration

### Prompts

The `proofreader_prompts.json` file contains the configurations for different types of feedback prompts. Each prompt group can be enabled or disabled using the `useprompt` field.

### Cases

The `cases_template.json` file serves as a template for defining cases. Each case includes an ID, title, description, solution, and difficulty level.
