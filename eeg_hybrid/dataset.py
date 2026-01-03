import typer
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import numpy as np
import moabb
from moabb.datasets import BNCI2014001, BNCI2014004
from mne.filter import filter_data

# Import paths from your auto-generated config
from eeg_hybrid.config import RAW_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    dataset_name: str = "2a",  # Choice of '2a' or '2b'
    # Mu and beta bands chosen pertaining to Event-Related Desynchronization (ERD)
    #   during MI; when a subject imagines moving, the power in these bands
    #   drops over the motor cortex
    low_hz: float = 8.0,       # Roadmap: Mu/Beta rhythms
    high_hz: float = 30.0,
):
    """
    Stage 1: Data Ingestion & Preprocessing.
    Fetches BCIC datasets via MOABB and applies standard MI filters.
    """
    logger.info(f"Initializing Stage 1 ingestion for BCIC IV {dataset_name}...")

    # Select dataset based on Phase 1 plan
    if dataset_name == "2a":
        ds = BNCI2014001()  # 4-class motor imagery
    else:
        ds = BNCI2014004()  # Binary motor imagery

    # MOABB fetches data in a single line as requested
    logger.info("Fetching data from MOABB (this may take a while on first run)...")
    # Fetches ALL subjects for the chosen dataset (9 for 2a, 9 for 2b)
    subjects = ds.subject_list 
    ds.download(subject_list=subjects, path=str(RAW_DATA_DIR))

    # This method does NOT take a path; it automatically searches 
    # standard locations and the path you just used in download()
    data_dict = ds.get_data(subjects=subjects)

    logger.info(f"Applying filters ({low_hz}-{high_hz} Hz) and Normalization...")
    
    for subject, sessions in tqdm(data_dict.items(), desc="Subjects"):
        for session, runs in sessions.items():
            subject_dir = INTERIM_DATA_DIR / dataset_name / f"subject_{subject}" / session
            subject_dir.mkdir(parents=True, exist_ok=True)
            for run, raw in runs.items():
                
                # 1. Bandpass and Notch Filter (Roadmap Stage 1)
                # Bandpass explained above, firwin parameter uses a Finite Impulse Response
                #   (FIR), a linear-phase filter delaying all frequencies by the same amount
                #   to prevent phase distortion to avoid smearing temporal relationships
                #   per each trial.
                # Notch filter removes line noise from electrical grid, which can be significant
                #   compared to brain signals.
                #   Data gathered from Austria where grid runs at 50Hz.
                raw.filter(l_freq=low_hz, h_freq=high_hz, fir_design='firwin', verbose=False)
                raw.notch_filter(freqs=[50.0], verbose=False) 
                
                # 2. Z-score Normalization (Roadmap Phase 1)
                # Using axis=1 calculates a unique mean and SD for each channel independently
                # Z-score normalization fixes power variance and inter-subject varability;
                #   also helps NN converge faster by removing threat of vanishing gradients;
                #   1e-8 avoids div by 0
                data = raw.get_data()
                mean = np.mean(data, axis=1, keepdims=True)
                std = np.std(data, axis=1, keepdims=True)
                # Avoid division by zero
                normalized_data = (data - mean) / (std + 1e-8)

                # --- SLIDING WINDOW (CROPPED TRAINING) ---
                # Parameters based on roadmap: 2s windows, 0.5s overlap 
                sfreq = raw.info['sfreq']  # Sampling frequency (e.g., 250Hz for BCIC IV 2a)
                window_size_samples = int(2.0 * sfreq) 
                stride_samples = int(0.5 * sfreq)

                # Calculate the number of windows possible in the 4s trial
                num_windows = (normalized_data.shape[1] - window_size_samples) // stride_samples + 1

                for i in range(num_windows):
                    start = i * stride_samples
                    end = start + window_size_samples
                    window_data = normalized_data[:, start:end]
    
                    # SANITY CHECK: Verify dimensions and timing
                    # Expected for 2a: (22, 500)
                    logger.debug(f"Window {i}: Start Sample {start}, End Sample {end}")
                    logger.debug(f"Window {i} Shape: {window_data.shape}")
    
                    # Ensure the window is exactly the right length before saving
                    assert window_data.shape[1] == window_size_samples, f"Error: Window {i} is wrong size!"
                    
                    # Update saving logic to include window index
                    output_name = f"{run}_win{i}_cleaned.npy"
                    save_path = subject_dir / output_name
                    np.save(save_path, window_data)

    logger.success(f"Stage 1 complete. Cleaned EEG saved to {INTERIM_DATA_DIR}")

if __name__ == "__main__":
    app()