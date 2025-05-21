# This is the main file for the FastAPI application.
# It will contain the API endpoints and logic for the visualizer.

import uvicorn
import os # Added import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Corrected absolute import assuming main.py is in visualizer_app and data_loader.py is in the same directory
from data_loader import VisualizationDataLoader 

# Define absolute paths for static and templates directories
MAIN_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR_ABSOLUTE = os.path.join(MAIN_SCRIPT_DIR, "static")
TEMPLATES_DIR_ABSOLUTE = os.path.join(MAIN_SCRIPT_DIR, "templates")

app = FastAPI()

# Mount static files using absolute path
app.mount("/static", StaticFiles(directory=STATIC_DIR_ABSOLUTE), name="static")
# Initialize Jinja2Templates using absolute path
templates = Jinja2Templates(directory=TEMPLATES_DIR_ABSOLUTE)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/sequence")
async def get_sequence_data(
    resolution: int = 128,
    pacman_sequence_length: int = 8 # PacmanDataset's sequence_length is L.
                                      # The visualizer will display L-1 frames.
):
    try:
        # Instantiate VisualizationDataLoader with the new parameters
        loader = VisualizationDataLoader(
            resolution=resolution,
            pacman_sequence_length=pacman_sequence_length
        )
        # get_visualization_data now takes no parameters
        viz_data = loader.get_visualization_data() 
        
        print(f"[MAIN_API] Data from loader (length): {len(viz_data)}")
        if viz_data:
            # print(f"[MAIN_API] First item from loader: {viz_data[0]}")
            print(f"[MAIN_API] First item from loader: {{'frame_url': '{viz_data[0]['frame_url']}', 'actions_length': len(viz_data[0]['actions']), 'original_frame_index_in_raw': {viz_data[0]['original_frame_index_in_raw']}}}")

        return {"sequence_data": viz_data}
    except Exception as e:
        # Basic error handling for now
        # In a real app, you might want to log the error and return a more specific HTTP error
        return {"error": str(e)}, 500

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
