// Main application JavaScript for Pacman Dataset Visualization

document.addEventListener('DOMContentLoaded', () => {
    // DOM elements
    const initDatasetBtn = document.getElementById('initDataset');
    const nextSequenceBtn = document.getElementById('nextSequence');
    const toggleCacheBtn = document.getElementById('toggleCache');
    const sequenceContainer = document.getElementById('sequenceContainer');
    const cacheContainer = document.getElementById('cacheContainer');
    const cacheList = document.getElementById('cacheList');
    const loadingIndicator = document.getElementById('loadingIndicator');
    
    // Templates
    const sequenceTemplate = document.getElementById('sequenceTemplate');
    const frameTemplate = document.getElementById('frameTemplate');
    
    // State
    let isDatasetInitialized = false;
    let isCacheVisible = false;
    let currentSequence = null; // Store the current sequence data
    
    // Action labels mapping (swapped LEFT and RIGHT to match backend)
    const ACTION_LABELS = ["LEFT", "RIGHT", "UP", "DOWN", "NO_ACTION"];
    
    // Action colors mapping
    const ACTION_COLORS = {
        'LEFT': '#3498db',    // Blue (was RIGHT)
        'RIGHT': '#e74c3c',   // Red (was LEFT)
        'UP': '#2ecc71',      // Green
        'DOWN': '#f39c12',    // Orange
        'NO_ACTION': '#95a5a6' // Gray
    };
    
    // Stagger offset controls
    const staggerOffsetSlider = document.getElementById('staggerOffset');
    const staggerOffsetValue = document.getElementById('staggerOffsetValue');
    const decrementBtn = document.getElementById('decrementOffset');
    const incrementBtn = document.getElementById('incrementOffset');
    
    /**
     * Compute action labels based on action indices and current stagger offset
     * @param {Array} indices - Array of action indices
     * @param {Number} offset - Stagger offset value
     * @param {Object} displayRange - Information about the display range
     * @returns {Array} - Array of objects with name and text properties
     */
    function computeLabels(indices, offset, displayRange) {
        // Safety check for undefined indices
        if (!indices || !Array.isArray(indices)) {
            console.error('Invalid action indices:', indices);
            return [];
        }
        
        // Default display range if not provided (backward compatibility)
        const range = displayRange || {
            start: 0,
            end: indices.length,
            total: indices.length
        };
        
        // Create labels for the frames we're displaying
        const labels = [];
        
        // For each frame in our display range
        for (let displayIdx = 0; displayIdx < (range.end - range.start); displayIdx++) {
            // Calculate the actual frame index in the full sequence
            const frameIdx = displayIdx + range.start;
            
            // Calculate the action index with offset
            const actionIdx = frameIdx - offset;
            
            let name, text;
            
            // Check if the action index is within the valid range of all actions
            if (0 <= actionIdx && actionIdx < indices.length) {
                // Get the action at this index
                const actionValue = indices[actionIdx];
                
                // Handle different data types (backward compatibility)
                if (typeof actionValue === 'number') {
                    // Numeric index into ACTION_LABELS
                    name = ACTION_LABELS[actionValue] || "Unknown";
                } else if (typeof actionValue === 'string' && !isNaN(parseInt(actionValue))) {
                    // String that can be parsed as a number
                    name = ACTION_LABELS[parseInt(actionValue)] || "Unknown";
                } else {
                    // Direct action name
                    name = actionValue;
                }
                
                // Format the text based on offset
                if (offset > 0) {
                    text = `${name} (from ${offset} frame${offset !== 1 ? 's' : ''} ago)`;
                } else if (offset < 0) {
                    text = `${name} (in ${-offset} frame${offset !== -1 ? 's' : ''} time)`;
                } else {
                    text = `${name} (current frame)`;
                }
            } else {
                // This should rarely happen now with our extended context
                name = "NO_ACTION";
                text = offset > 0 ? "No past action" : "Future action unknown";
                console.warn(`Action out of bounds: frame ${frameIdx}, offset ${offset}, action index ${actionIdx}`);
            }
            
            labels.push({ name, text });
        }
        
        return labels;
    }
    
    // Update the display value and re-render if needed
    function updateStaggerDisplay(value) {
        const numValue = parseInt(value);
        staggerOffsetValue.textContent = numValue >= 0 ? `+${numValue}` : numValue;
        staggerOffsetSlider.value = numValue;
        
        // Re-render the current sequence with the new offset if available
        if (currentSequence) {
            renderSequence(currentSequence, numValue);
        }
    }
    
    // Handle slider input
    staggerOffsetSlider.addEventListener('input', (e) => {
        updateStaggerDisplay(e.target.value);
    });
    
    // Handle increment/decrement buttons
    decrementBtn.addEventListener('click', () => {
        const newValue = Math.max(-5, parseInt(staggerOffsetSlider.value) - 1);
        updateStaggerDisplay(newValue);
    });
    
    incrementBtn.addEventListener('click', () => {
        const newValue = Math.min(5, parseInt(staggerOffsetSlider.value) + 1);
        updateStaggerDisplay(newValue);
    });
    
    // Initialize display
    updateStaggerDisplay(staggerOffsetSlider.value);
    
    // Initialize dataset with parameters from form
    initDatasetBtn.addEventListener('click', async () => {
        const sequenceLength = parseInt(document.getElementById('sequenceLength').value);
        const resolution = parseInt(document.getElementById('resolution').value);
        const staggerOffset = parseInt(document.getElementById('staggerOffset').value);
        
        if (isNaN(sequenceLength) || isNaN(resolution) || isNaN(staggerOffset)) {
            alert('Please enter valid numbers for all fields.');
            return;
        }
        
        showLoading(true);
        
        try {
            const response = await fetch('/api/init', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    sequence_length: sequenceLength, 
                    resolution: resolution,
                    stagger_offset: staggerOffset
                })
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                isDatasetInitialized = true;
                nextSequenceBtn.disabled = false;
                alert('Dataset initialized successfully!');
            } else {
                alert(`Failed to initialize dataset: ${result.message}`);
            }
        } catch (error) {
            console.error('Error initializing dataset:', error);
            alert('Failed to initialize dataset. Check console for details.');
        } finally {
            showLoading(false);
        }
    });
    
    // Fetch and display the next sequence
    nextSequenceBtn.addEventListener('click', async () => {
        if (!isDatasetInitialized) {
            alert('Please initialize the dataset first!');
            return;
        }
        
        await fetchNextSequence();
    });
    
    // Toggle cache view
    toggleCacheBtn.addEventListener('click', async () => {
        isCacheVisible = !isCacheVisible;
        
        if (isCacheVisible) {
            await loadCachedSequences();
            cacheContainer.style.display = 'block';
            toggleCacheBtn.textContent = 'Hide Cache';
        } else {
            cacheContainer.style.display = 'none';
            toggleCacheBtn.textContent = 'Show Cache';
        }
    });
    
    // Fetch the next sequence from the API
    async function fetchNextSequence() {
        showLoading(true);
        
        try {
            const response = await fetch('/api/next_sequence');
            const result = await response.json();
            
            if (result.status === 'success') {
                displaySequence(result.data);
                await loadCachedSequences();  // Refresh cache list
            } else {
                alert(`Failed to fetch sequence: ${result.message}`);
            }
        } catch (error) {
            console.error('Error fetching sequence:', error);
            alert('Failed to fetch sequence. Check console for details.');
        } finally {
            showLoading(false);
        }
    }
    
    // Load and display cached sequences
    async function loadCachedSequences() {
        showLoading(true);
        
        try {
            const response = await fetch('/api/cached_sequences');
            const result = await response.json();
            
            if (result.status === 'success') {
                displayCachedSequences(result.sequences);
            } else {
                alert(`Failed to load cached sequences: ${result.message}`);
            }
        } catch (error) {
            console.error('Error loading cached sequences:', error);
            alert('Failed to load cached sequences. Check console for details.');
        } finally {
            showLoading(false);
        }
    }
    
    // Display a specific cached sequence
    async function loadCachedSequence(sequenceId) {
        showLoading(true);
        
        try {
            const response = await fetch(`/api/sequence/${sequenceId}`);
            const result = await response.json();
            
            if (result.status === 'success') {
                displaySequence(result.data);
            } else {
                alert(`Failed to load cached sequence: ${result.message}`);
            }
        } catch (error) {
            console.error('Error loading cached sequence:', error);
            alert('Failed to load cached sequence. Check console for details.');
        } finally {
            showLoading(false);
        }
    }
    
    /**
     * Render a sequence with the specified stagger offset
     * @param {Object} sequenceData - The sequence data with frames and action indices
     * @param {Number} offset - The stagger offset to apply
     */
    function renderSequence(sequenceData, offset) {
        // Clear previous content
        sequenceContainer.innerHTML = '';
        
        // Clone the sequence template
        const sequenceNode = sequenceTemplate.content.cloneNode(true);
        
        // Get the frames container
        const framesContainer = sequenceNode.querySelector('.frames-container');
        
        // Handle both old format (actions) and new format (action_indices)
        let actionData;
        if (sequenceData.action_indices) {
            // New format
            actionData = sequenceData.action_indices;
        } else if (sequenceData.actions) {
            // Old format - already formatted strings
            // We'll extract the action names from the strings
            actionData = sequenceData.actions.map(actionStr => {
                // Extract just the action name (e.g., "LEFT" from "LEFT (from 1 frame ago)")
                return actionStr.split(' ')[0];
            });
        } else {
            console.error('No action data found in sequence:', sequenceData);
            actionData = [];
        }
        
        // Compute action labels with the current offset and display range
        const actionLabels = computeLabels(actionData, offset, sequenceData.display_range);
        
        // Add sequence info if available
        if (sequenceData.display_range) {
            const { start, end, total, target_idx } = sequenceData.display_range;
            const infoElement = document.createElement('div');
            infoElement.className = 'sequence-info alert alert-info';
            infoElement.innerHTML = `
                <small>
                    Displaying frames ${start+1}-${end+1} of ${total}<br>
                    Target frame is the next frame (${target_idx || (end+2)})
                </small>`;
            sequenceNode.querySelector('.sequence-title').after(infoElement);
        }
        
        // Add each frame
        sequenceData.frames_b64.forEach((frameBase64, index) => {
            const frameNode = frameTemplate.content.cloneNode(true);
            
            // Calculate the actual frame number in the full sequence if display_range is available
            let frameNumber = index + 1;
            if (sequenceData.display_range) {
                frameNumber = index + sequenceData.display_range.start + 1;
            }
            
            // Set frame number
            frameNode.querySelector('.frame-number').textContent = `Frame ${frameNumber}`;
            
            // Set frame image
            const frameImage = frameNode.querySelector('.frame-image');
            frameImage.src = `data:image/png;base64,${frameBase64}`;
            
            // Set action with color coding
            if (index < actionLabels.length) {
                const actionLabel = actionLabels[index];
                const actionElement = frameNode.querySelector('.frame-action');
                actionElement.textContent = actionLabel.text;
                
                // Set background color based on action
                const color = ACTION_COLORS[actionLabel.name] || '#95a5a6';
                actionElement.style.backgroundColor = color;
                actionElement.style.color = '#fff';
                actionElement.style.borderRadius = '4px';
                actionElement.style.padding = '2px 5px';
            }
            
            framesContainer.appendChild(frameNode);
        });
        
        // Set target image with label
        const targetImage = sequenceNode.querySelector('.target-image');
        targetImage.src = `data:image/png;base64,${sequenceData.target_b64}`;
        
        // Add target frame label if not already present
        const targetContainer = targetImage.closest('.target-container');
        if (targetContainer) {
            const targetLabel = targetContainer.querySelector('.target-label') || document.createElement('div');
            if (!targetContainer.querySelector('.target-label')) {
                targetLabel.className = 'target-label mt-2 text-center fw-bold';
                targetContainer.appendChild(targetLabel);
            }
            
            // Set the target frame number
            const targetIdx = sequenceData.display_range?.target_idx || 
                             (sequenceData.display_range?.end + 2) || 
                             'Next';
            targetLabel.textContent = `Target Frame (${targetIdx})`;
        }
        
        // Add the sequence to the container
        sequenceContainer.appendChild(sequenceNode);
    }
    
    /**
     * Display a sequence in the UI with the current stagger offset
     * @param {Object} sequenceData - The sequence data with frames and action indices
     */
    function displaySequence(sequenceData) {
        if (!sequenceData || !sequenceData.frames_b64) {
            console.error('Invalid sequence data:', sequenceData);
            alert('Error: Invalid sequence data received');
            return;
        }
        
        // Store the current sequence data for later re-rendering
        currentSequence = sequenceData;
        
        // Render the sequence with the current stagger offset
        const offset = parseInt(staggerOffsetSlider.value);
        renderSequence(sequenceData, offset);
    }
    
    // Display cached sequences in the UI
    function displayCachedSequences(sequences) {
        // Clear previous content
        cacheList.innerHTML = '';
        
        if (sequences.length === 0) {
            cacheList.innerHTML = '<p class="text-center col-12">No cached sequences found.</p>';
            return;
        }
        
        // Add each cached sequence
        sequences.forEach(sequence => {
            const col = document.createElement('div');
            col.className = 'col-md-3 mb-4';
            
            const cacheItem = document.createElement('div');
            cacheItem.className = 'cache-item';
            cacheItem.addEventListener('click', () => loadCachedSequence(sequence.id));
            
            const thumbnail = document.createElement('img');
            thumbnail.className = 'cache-thumbnail';
            thumbnail.src = `data:image/png;base64,${sequence.preview}`;
            thumbnail.alt = 'Sequence Preview';
            
            const title = document.createElement('div');
            title.className = 'cache-item-title';
            title.textContent = `Sequence ${sequence.id.substring(0, 8)}...`;
            
            cacheItem.appendChild(thumbnail);
            cacheItem.appendChild(title);
            col.appendChild(cacheItem);
            cacheList.appendChild(col);
        });
    }
    
    // Show/hide loading indicator
    function showLoading(isLoading) {
        loadingIndicator.style.display = isLoading ? 'block' : 'none';
        initDatasetBtn.disabled = isLoading;
        nextSequenceBtn.disabled = isLoading || !isDatasetInitialized;
        toggleCacheBtn.disabled = isLoading;
    }
    
    // Initialize the UI
    function initUI() {
        nextSequenceBtn.disabled = true;
    }
    
    // Start the application
    initUI();
});
