# Systematic Debugging Methodology for Deep Learning Models

## Problem: Tensor Shape Mismatches in Complex Model Architectures

### Systematic Approach

1. **Trace Data Flow**
   - Identify the critical path of tensor transformations
   - Add strategic debug prints at key transformation points
   - Focus on shape changes between components

2. **Component Interaction Analysis**
   - Identify where different components interact (e.g., VAE, DPM-Solver, main model)
   - Check if assumptions about tensor shapes hold at boundaries
   - Pay special attention to batch dimensions and conditioning

3. **Minimal Debugging Strategy**
   ```python
   print(f"=== Component Name Debug ===")
   print(f"Input shapes: {x.shape}, {y.shape}")
   print(f"Output shape: {output.shape}")
   ```

4. **Batch Size Consistency**
   - Track batch size changes, especially with:
     - Classifier-free guidance (doubles batch size)
     - Conditional inputs
     - Model kwargs

5. **Resolution Validation**
   - Verify spatial dimensions match at concatenation points
   - Check encoder/decoder expectations
   - Validate against model architecture requirements

### Case Study: DPM-Solver Batch Size Mismatch

1. **Problem Identification**
   - Observed tensor dimension mismatch during validation
   - Traced to classifier-free guidance doubling some but not all tensors

2. **Root Cause Analysis**
   - DPM-Solver doubled `x`, `t`, and condition tensors
   - Additional inputs (`obs`, `x_pixel_space`) weren't doubled
   - Led to inconsistent batch sizes

3. **Solution Strategy**
   - Identified specific tensors requiring doubling
   - Modified DPM-Solver wrapper to handle additional inputs
   - Maintained minimal changes principle
   - Verified solution with debug prints

4. **Validation**
   - Confirmed all tensor shapes match
   - Verified model runs without errors
   - Checked output quality

### Best Practices

1. **Minimal Changes**
   - Only modify what's necessary
   - Document why each change is needed
   - Keep original functionality intact

2. **Debug Print Hygiene**
   - Add prints strategically
   - Remove unnecessary debugging once fixed
   - Keep critical shape checks for future reference

3. **Error Prevention**
   - Validate shapes before expensive operations
   - Check assumptions at component boundaries
   - Consider edge cases (batch size changes, different modes)
