# This is the main file for the FastAPI application.
# It will contain the API endpoints and logic for the visualizer.

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Corrected relative import assuming main.py is in visualizer_app and data_loader.py is in the same directory
from .data_loader import VisualizationDataLoader 

app = FastAPI()

# Assuming running from project root, so paths are relative to the root.
templates = Jinja2Templates(directory="visualizer_app/templates")
app.mount("/static", StaticFiles(directory="visualizer_app/static"), name="static")

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
        return {"sequence_data": viz_data}
    except Exception as e:
        # Basic error handling for now
        # In a real app, you might want to log the error and return a more specific HTTP error
        return {"error": str(e)}, 500

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
