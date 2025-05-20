// This is the main JavaScript file.

let currentRawData = []; // To store the 'sequence_data' array from the backend
const ACTION_COLORS = {"LEFT": "red", "RIGHT": "blue", "UP": "green", "DOWN": "purple", "NO_ACTION": "grey", "N/A": "black"};

// DOM Elements
const loadDataBtn = document.getElementById('loadDataBtn');
const staggerOffsetInput = document.getElementById('staggerOffset');
const visualizationContainer = document.getElementById('visualizationContainer');

function updateActionDisplay() {
    const offset = parseInt(staggerOffsetInput.value) || 0;
    const frameElements = visualizationContainer.children;

    for (let i = 0; i < frameElements.length; i++) {
        const frameDiv = frameElements[i];
        
        // Retrieve original_frame_index_in_raw from dataset.
        // As set in renderSequence, this is Math.floor(item.actions.length / 2),
        // which is the index of the action for offset 0 within the per-frame actions array.
        const baseOriginalFrameIndex = parseInt(frameDiv.dataset.originalFrameIndexInRaw);
        
        // Retrieve the full actions array (per-frame context window) from dataset.
        const actions = JSON.parse(frameDiv.dataset.actions); 
        
        // Calculate the index of the action to display using the formula from the subtask.
        const action_idx_to_display = baseOriginalFrameIndex - offset;
        
        let actionText = "N/A";
        if (actions && action_idx_to_display >= 0 && action_idx_to_display < actions.length) {
            actionText = actions[action_idx_to_display];
        }

        const actionLabel = frameDiv.querySelector('p.action-label');
        if (actionLabel) {
            actionLabel.textContent = actionText;
            actionLabel.style.color = ACTION_COLORS[actionText] || ACTION_COLORS["N/A"];
        }
    }
}

function renderSequence(sequence) {
    currentRawData = sequence; // Store the raw data
    visualizationContainer.innerHTML = ''; // Clear previous content

    if (!currentRawData || currentRawData.length === 0) {
        visualizationContainer.innerHTML = '<p>No data to display.</p>';
        return;
    }

    currentRawData.forEach((item, index) => {
        const frameDiv = document.createElement('div');
        frameDiv.classList.add('frame-item');
        
        // Store necessary data on the element for updateActionDisplay
        // item.actions is an array of action strings for the context window of this frame.
        // Its length is 2 * contextWindowHalf + 1.
        // The action corresponding to the frame itself (offset 0) is at the center of this array.
        frameDiv.dataset.actions = JSON.stringify(item.actions); 

        // item.actions is expected to be the full, shared list of action strings for the entire context window.
        frameDiv.dataset.actions = JSON.stringify(item.actions); 

        // item.original_frame_index_in_raw (from backend) is the index in the shared 'item.actions' array
        // that corresponds to this frame's action when the UI offset is 0.
        // This value is stored directly.
        frameDiv.dataset.originalFrameIndexInRaw = item.original_frame_index_in_raw;

        const img = document.createElement('img');
        img.src = item.frame_url;
        img.alt = `Frame ${index}`;

        const actionLabel = document.createElement('p');
        actionLabel.classList.add('action-label');
        // Initial text will be set by updateActionDisplay

        frameDiv.appendChild(img);
        frameDiv.appendChild(actionLabel);
        visualizationContainer.appendChild(frameDiv);
    });

    updateActionDisplay(); // Set initial actions based on the current offset
}

async function fetchAndDisplaySequence() {
    // Default parameters for the API request, matching backend defaults
    const resolution = 128;
    const pacman_sequence_length = 8; // PacmanDataset's sequence_length is L.
                                       // The visualizer will display L-1 frames.

    // Construct the API URL with the new parameters
    const apiUrl = `/api/sequence?resolution=${resolution}&pacman_sequence_length=${pacman_sequence_length}`;

    try {
        const response = await fetch(apiUrl);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const jsonData = await response.json();
        if (jsonData.error) {
            throw new Error(`Backend error: ${jsonData.error}`);
        }
        if (jsonData.sequence_data) {
            renderSequence(jsonData.sequence_data);
        } else {
            console.error("No sequence_data found in response:", jsonData);
            visualizationContainer.innerHTML = '<p>Error: No sequence data received.</p>';
        }
    } catch (error) {
        console.error('Error fetching or processing sequence data:', error);
        visualizationContainer.innerHTML = `<p>Error loading data: ${error.message}. Check console for details.</p>`;
    }
}

// Event Listeners
if (loadDataBtn) {
    loadDataBtn.addEventListener('click', fetchAndDisplaySequence);
}

if (staggerOffsetInput) {
    staggerOffsetInput.addEventListener('input', updateActionDisplay);
}

// Optional: Load initial data when the page loads
// document.addEventListener('DOMContentLoaded', fetchAndDisplaySequence);
// For now, relying on button click as per instructions.
