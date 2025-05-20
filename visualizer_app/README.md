# Frame Action Visualizer

This web application provides an interactive interface to visualize game state frames and corresponding agent actions, sourced from the Pacman dataset. It helps in inspecting and debugging the data that is fed into a model during training.

## Prerequisites

*   Python 3.7+
*   Access to a terminal or command prompt.

## Setup Instructions

1.  **Clone the Repository (if you haven't already):**
    ```bash
    # If the visualizer_app is part of a larger repository
    git clone <repository_url>
    cd <repository_root>/visualizer_app 
    # If you have just the visualizer_app directory, navigate into it
    # cd visualizer_app
    ```

2.  **Create a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    *   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install Dependencies:**
    Ensure you are in the `visualizer_app` directory where `requirements.txt` is located.
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  **Start the FastAPI Server:**
    Make sure your virtual environment is activated and you are in the `visualizer_app` directory.
    Run the main application file:
    ```bash
    python main.py
    ```
    You should see output indicating the Uvicorn server is running, typically on `http://127.0.0.1:8000`.

2.  **Access the Visualizer:**
    Open your web browser and navigate to:
    [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Using the Visualizer

*   **Load Sequence Data:** Click the "Load Sequence Data" button to fetch and display a sequence of frames and actions. Each click will load the next available sequence from the dataset.
*   **Stagger Offset:** Use the "Stagger Offset (N)" input field to change the alignment between frames and actions:
    *   **0:** Frame `i` is shown with action `action[i]`.
    *   **+N (positive):** Frame `i` is shown with `action[i-N]` (the action that occurred N steps *before* frame `i`).
    *   **-N (negative):** Frame `i` is shown with `action[i+N]` (the action that will occur N steps *after* frame `i`).
    The action labels will update in real-time as you change the offset.
*   **Action Colors:** Actions are color-coded for easier visual distinction.
