import numpy as np

def inspect_large_npz(file_path, num_samples=1):
    """
    Safely inspects and prints a preview of a very large .npz file 
    without loading it entirely into RAM.
    """
    print(f"🔍 Opening '{file_path}' using memory-mapping...\n")
    
    # mmap_mode='r' keeps the data on disk and only reads what you ask to print
    with np.load(file_path, mmap_mode='r') as data:
        
        # 1. List all available keys (arrays) inside the file
        keys = data.files
        print(f"📦 Arrays found in archive: {keys}")
        print("=" * 70)
        
        for key in keys:
            arr = data[key]
            
            # 2. Print metadata for the current array
            print(f"👉 KEY NAME: '{key}'")
            print(f"   • Array Shape: {arr.shape}")
            print(f"   • Data Type:   {arr.dtype}")
            print(f"   • Dimensions:  {arr.ndim}")
            print(f"   • Total Items: {arr.size:,}")
            print("-" * 50)
            
            # 3. Smart slicing based on array dimensions to keep terminal clean
            print(f"   📄 Previewing the first {num_samples} sample(s):")
            
            if arr.ndim == 1:
                # 1D Array (e.g., your labels array)
                print(f"   {arr[:num_samples]}")
                
            elif arr.ndim == 2:
                # 2D Array
                print(arr[:num_samples, :])
                
            elif arr.ndim == 3:
                # 3D Array (like your telemetry data: [N, 160, 20])
                # Printing all 160 timesteps is too much, so we show a snapshot
                for i in range(min(num_samples, arr.shape[0])):
                    print(f"\n   [Sample {i}] Snapshot (First 5 timesteps, first 5 features):")
                    # Slice a 5x5 window out of the 160x20 sample
                    snapshot = arr[i, :5, :5]
                    print(snapshot)
                    print("   ... [truncated for readability] ...")
            else:
                # Fallback for higher dimensions
                print(arr[:num_samples])
                
            print("=" * 70 + "\n")

# ──────────────────────────────────────────────
# Run the inspector on your specific dataset
# ──────────────────────────────────────────────
inspect_large_npz('DASHlink_full_fourclass_raw_comp.npz', num_samples=1)