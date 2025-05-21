// This is the main JavaScript file.

let currentRawData = []; // To store the 'sequence_data' array from the backend
const ACTION_COLORS = {"LEFT": "red", "RIGHT": "blue", "UP": "green", "DOWN": "purple", "NO_ACTION": "grey", "N/A": "black"};

// DOM Elements
const loadDataBtn = document.getElementById('loadDataBtn');
const staggerOffsetInput = document.getElementById('staggerOffset');
const visualizationContainer = document.getElementById('visualizationContainer');

function updateActionDisplay() {
    console.log('[SCRIPT_JS] updateActionDisplay called. Current offset:', staggerOffsetInput.value);
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

        // Debugging for the first frame element
        if (i === 0 && frameDiv.dataset) {
            console.log('[SCRIPT_JS] Updating first frame. Stored originalFrameIndexInRaw:', frameDiv.dataset.originalFrameIndexInRaw);
            let actions_for_log;
            try {
                actions_for_log = JSON.parse(frameDiv.dataset.actions); // This is 'actions' variable from above
            } catch(e) {
                actions_for_log = "Error parsing actions";
            }
            // Check if actions_for_log is an array before slicing
            const actionsSample = Array.isArray(actions_for_log) ? actions_for_log.slice(0,5).join(", ") + "..." : actions_for_log;
            console.log('[SCRIPT_JS] Updating first frame. Stored actions (sample):', actionsSample);
            
            console.log('[SCRIPT_JS] For first frame: baseOriginalFrameIndex:', baseOriginalFrameIndex, 'offset:', offset, 'action_idx_to_display:', action_idx_to_display);
            
            // actionText is already calculated above, use it directly
            console.log('[SCRIPT_JS] For first frame: resulting actionText:', actionText);
        }

        const actionLabel = frameDiv.querySelector('p.action-label');
        if (actionLabel) {
            actionLabel.textContent = actionText;
            actionLabel.style.color = ACTION_COLORS[actionText] || ACTION_COLORS["N/A"];
        }
    }
}

function renderSequence(sequence) {
    console.log('[SCRIPT_JS] renderSequence called with sequence length:', sequence ? sequence.length : 0);
    currentRawData = sequence; // Store the raw data
    visualizationContainer.innerHTML = ''; // Clear previous content

    if (!currentRawData || currentRawData.length === 0) {
        visualizationContainer.innerHTML = '<p>No data to display.</p>';
        return;
    }

    currentRawData.forEach((item, index) => {
        if (index === 0) {
            console.log('[SCRIPT_JS] Processing first item in renderSequence:', JSON.parse(JSON.stringify(item)));
            console.log('[SCRIPT_JS] First item frame_url for img src:', item.frame_url);
        }

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

        // Debugging logs for received data
        console.log('[SCRIPT_JS] Received data from backend (raw jsonData):', JSON.parse(JSON.stringify(jsonData))); 
        if (jsonData && jsonData.sequence_data) {
            console.log('[SCRIPT_JS] sequence_data length:', jsonData.sequence_data.length);
            if (jsonData.sequence_data.length > 0) {
                console.log('[SCRIPT_JS] First item of sequence_data:', JSON.parse(JSON.stringify(jsonData.sequence_data[0])));
            }
            renderSequence(jsonData.sequence_data); // Call renderSequence if data is valid
        } else {
            console.error('[SCRIPT_JS] jsonData.sequence_data is missing or undefined', jsonData);
            // Also handle error if jsonData.error is present (as it was before)
            if (jsonData.error) {
                 throw new Error(`Backend error: ${jsonData.error}`);
            }
            visualizationContainer.innerHTML = '<p>Error: No sequence data received or data is malformed.</p>';
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
