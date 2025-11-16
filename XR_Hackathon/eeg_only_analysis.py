"""
EEG-ONLY ANALYSIS SCRIPT FOR MOTOR CONTROL REHABILITATION
Complete pipeline: Load data → Find markers → Extract biomarkers → Feedback

Analyzes:
- Attention/Focus levels (Beta/Theta ratio)
- Motor Control engagement (Beta/Alpha ratio, Mu suppression)
- Baseline vs Task comparison
- Provides rehabilitation feedback

Data Format: Single file with markers
EEG Channels: Ch1-8
Markers: 1st & 2nd = baseline, 3rd & 4th = task

Author: Claude
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
try:
    from scipy.integrate import simpson
except ImportError:
    from scipy.integrate import simps as simpson
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


# =============================================================================
# CONFIGURATION - EDIT THIS SECTION
# =============================================================================

# Your data file path
DATA_FILE = "/Users/joonan/Desktop/OpenBCI_GUI/Recordings/OpenBCISession_2025-11-16_05-44-45/OpenBCI-RAW-2025-11-16_05-49-45.txt"

# Task/Exercise name (for feedback context)
TASK_NAME = "Motor Control Exercise"  # e.g., "Reaching Task", "Grasping Exercise"

# Marker configuration
# Your markers are labeled 1, 2, 3, 4
# Baseline between markers 1 and 2
# Task between markers 3 and 4
BASELINE_MARKERS = [1, 2]  # Baseline between markers 1 and 2
TASK_MARKERS = [3, 4]      # Task between markers 3 and 4

# Settings
SAMPLING_RATE = 125        # Hz (OpenBCI default)
SAVE_FIGURES = True        # Save PNG files?
SHOW_PLOTS = False         # Display plots on screen?
OUTPUT_DPI = 300           # Image resolution


# =============================================================================
# STEP 1: DATA LOADING
# =============================================================================

def load_data(filepath):
    """Load OpenBCI data file"""
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    print(f"File: {filepath}")
    
    try:
        # Read file to find where actual data starts
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # OpenBCI files have header lines starting with %
        # Find the last header line
        data_start_line = 0
        header_line = None
        
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                continue
            
            # Lines starting with % are header/comments
            if line.startswith('%'):
                data_start_line = i + 1
                # Check if this is the column header line
                if 'Sample Index' in line or 'EXG Channel' in line:
                    header_line = i
            # First line without % is likely the column headers (if not found yet)
            elif header_line is None and ('Sample' in line or 'Channel' in line or 'EXG' in line):
                header_line = i
                data_start_line = i
                break
        
        print(f"  Found header at line {header_line}")
        print(f"  Data starts at line {data_start_line}")
        
        # Try to load the data
        if header_line is not None:
            # Load with header
            df = pd.read_csv(filepath, skiprows=range(header_line), 
                           on_bad_lines='skip', encoding='utf-8', 
                           encoding_errors='ignore')
        else:
            # No header found, skip comment lines
            df = pd.read_csv(filepath, skiprows=data_start_line,
                           on_bad_lines='skip', encoding='utf-8',
                           encoding_errors='ignore')
        
        # Clean column names
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace('%', '')
        df.columns = df.columns.str.strip()
        
        # Remove any completely empty rows
        df = df.dropna(how='all')
        
        # Try to convert numeric columns
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
        
        print(f"✓ Loaded {len(df)} samples")
        print(f"✓ Duration: {len(df)/SAMPLING_RATE:.1f} seconds")
        print(f"✓ Columns: {list(df.columns[:5])}... ({len(df.columns)} total)")
        
        return df
        
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        print("\nTrying alternative loading method...")
        
        try:
            # Alternative: Try loading with different parameters
            df = pd.read_csv(filepath, skiprows=5, on_bad_lines='skip',
                           low_memory=False)
            df.columns = df.columns.str.strip()
            print(f"✓ Loaded {len(df)} samples using alternative method")
            return df
        except Exception as e2:
            print(f"✗ Alternative method also failed: {e2}")
            
            # Last resort: manual parsing
            print("\nAttempting manual parsing...")
            try:
                data_lines = []
                headers = None
                
                with open(filepath, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('%'):
                            if 'Sample Index' in line or 'EXG Channel' in line:
                                # This is the header
                                headers = line.replace('%', '').strip().split(',')
                                headers = [h.strip() for h in headers]
                            continue
                        
                        # Try to parse as data
                        parts = line.split(',')
                        if len(parts) > 1:  # Has multiple fields
                            data_lines.append(parts)
                
                if data_lines:
                    df = pd.DataFrame(data_lines, columns=headers if headers else None)
                    # Convert to numeric where possible
                    for col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='ignore')
                        except:
                            pass
                    
                    print(f"✓ Manually parsed {len(df)} samples")
                    return df
                else:
                    print("✗ Could not parse any data")
                    return None
                    
            except Exception as e3:
                print(f"✗ Manual parsing failed: {e3}")
                return None


# =============================================================================
# STEP 2: MARKER DETECTION
# =============================================================================

def find_markers(df):
    """Find markers in data"""
    print("\n" + "="*70)
    print("STEP 2: FINDING MARKERS")
    print("="*70)
    
    markers = {}  # Use dict with marker number as key
    marker_col = None
    
    # Look for marker column
    for col in ['Marker', 'marker', 'Event', 'event', 'Timestamp', 'timestamp']:
        if col in df.columns:
            marker_col = col
            break
    
    if marker_col:
        print(f"✓ Found marker column: '{marker_col}'")
        marker_rows = df[df[marker_col].notna() & (df[marker_col] != '')]
        
        for idx, row in marker_rows.iterrows():
            marker_value = row[marker_col]
            
            # Try to extract numeric marker value (1, 2, 3, 4)
            try:
                # If marker is like "1", "2", "3", "4" or 1, 2, 3, 4
                if isinstance(marker_value, (int, float)):
                    marker_num = int(marker_value)
                else:
                    # Try to extract number from string like "Marker 1" or "M1" or "1"
                    import re
                    numbers = re.findall(r'\d+', str(marker_value))
                    if numbers:
                        marker_num = int(numbers[0])
                    else:
                        # If no number found, use sequential numbering
                        marker_num = len(markers) + 1
            except:
                marker_num = len(markers) + 1
            
            markers[marker_num] = {
                'marker_number': marker_num,
                'sample': idx,
                'time': idx / SAMPLING_RATE,
                'label': str(marker_value)
            }
            print(f"  Marker {marker_num}: Sample {idx} ({idx/SAMPLING_RATE:.2f}s) - '{marker_value}'")
        
        print(f"✓ Total markers found: {len(markers)}")
    else:
        print("⚠ No marker column found - will use manual segmentation")
    
    return markers


# =============================================================================
# STEP 3: DATA SEGMENTATION
# =============================================================================

def segment_data(df, markers, baseline_markers, task_markers):
    """Segment into baseline and task periods"""
    print("\n" + "="*70)
    print("STEP 3: SEGMENTING DATA")
    print("="*70)
    
    segments = {}
    
    # Check if we have the required markers (markers is now a dict with marker numbers as keys)
    required_markers = set(baseline_markers + task_markers)
    
    if all(m in markers for m in required_markers):
        # Baseline (between marker 1 and marker 2)
        start = markers[baseline_markers[0]]['sample']
        end = markers[baseline_markers[1]]['sample']
        segments['baseline'] = df.iloc[start:end].copy()
        duration = (end - start) / SAMPLING_RATE
        print(f"✓ Baseline: Between markers {baseline_markers[0]} and {baseline_markers[1]}")
        print(f"  Samples {start}-{end} ({duration:.1f}s)")
        
        # Task (between marker 3 and marker 4)
        start = markers[task_markers[0]]['sample']
        end = markers[task_markers[1]]['sample']
        segments['task'] = df.iloc[start:end].copy()
        duration = (end - start) / SAMPLING_RATE
        print(f"✓ Task: Between markers {task_markers[0]} and {task_markers[1]}")
        print(f"  Samples {start}-{end} ({duration:.1f}s)")
    else:
        print(f"⚠ Required markers not found (need markers: {required_markers})")
        print("⚠ Using manual segmentation (first 30s baseline, 30s task at 60s)")
        baseline_samples = int(30 * SAMPLING_RATE)
        task_start = int(60 * SAMPLING_RATE)
        task_samples = int(30 * SAMPLING_RATE)
        
        segments['baseline'] = df.iloc[:baseline_samples].copy()
        segments['task'] = df.iloc[task_start:task_start + task_samples].copy()
        print(f"✓ Baseline: 0-30s")
        print(f"✓ Task: 60-90s")
    
    return segments


# =============================================================================
# STEP 4: EEG EXTRACTION AND PREPROCESSING
# =============================================================================

def extract_eeg(segment_df):
    """Extract EEG channels (Ch1-8)"""
    numeric_cols = segment_df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) >= 8:
        eeg_data = segment_df[numeric_cols[0:8]].values
    else:
        print(f"  ⚠ Warning: Expected 8 EEG channels, found {len(numeric_cols)}")
        eeg_data = segment_df[numeric_cols[:min(8, len(numeric_cols))]].values
    
    return eeg_data


def preprocess_eeg(eeg_data, fs=125):
    """Bandpass filter 0.5-50 Hz"""
    sos = signal.butter(4, [0.5, 50], btype='bandpass', fs=fs, output='sos')
    filtered = np.zeros_like(eeg_data)
    
    for i in range(eeg_data.shape[1]):
        filtered[:, i] = signal.sosfiltfilt(sos, eeg_data[:, i])
    
    return filtered


# =============================================================================
# STEP 5: BIOMARKER EXTRACTION
# =============================================================================

def calculate_bandpower(data, fs, band):
    """Calculate power in frequency band"""
    freqs, psd = signal.welch(data, fs=fs, nperseg=min(256, len(data)))
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return simpson(psd[idx], freqs[idx])


def extract_biomarkers(eeg_data, fs=125):
    """
    Extract attention and motor control biomarkers
    
    Returns:
    --------
    biomarkers : dict
        All EEG biomarkers with interpretations
    """
    print("\nExtracting biomarkers...")
    
    # Frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),    # Includes mu rhythm (8-12 Hz)
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    # Per-channel analysis
    attention_indices = []
    motor_control_indices = []
    mu_suppression = []
    
    band_powers = {band: [] for band in bands.keys()}
    
    for ch in range(eeg_data.shape[1]):
        data = eeg_data[:, ch]
        
        # Calculate all band powers
        powers = {}
        for band_name, band_range in bands.items():
            power = calculate_bandpower(data, fs, band_range)
            powers[band_name] = power
            band_powers[band_name].append(power)
        
        # Attention index (Beta/Theta ratio)
        # Higher = better attention/focus
        attention_idx = powers['beta'] / (powers['theta'] + 1e-10)
        attention_indices.append(attention_idx)
        
        # Motor control index (Beta/Alpha ratio)
        # Higher = more motor engagement
        motor_idx = powers['beta'] / (powers['alpha'] + 1e-10)
        motor_control_indices.append(motor_idx)
        
        # Mu suppression (inverse of alpha power)
        # Lower alpha during motor tasks = mu suppression = motor preparation
        mu_suppression.append(1.0 / (powers['alpha'] + 1e-10))
    
    # Aggregate metrics
    biomarkers = {
        # Attention metrics
        'attention_index': np.mean(attention_indices),
        'attention_std': np.std(attention_indices),
        'attention_per_channel': attention_indices,
        
        # Motor control metrics
        'motor_control_index': np.mean(motor_control_indices),
        'motor_control_std': np.std(motor_control_indices),
        'motor_per_channel': motor_control_indices,
        
        # Mu suppression (motor preparation)
        'mu_suppression': np.mean(mu_suppression),
        'mu_suppression_per_channel': mu_suppression,
        
        # Band powers
        'band_powers': {band: np.mean(powers) for band, powers in band_powers.items()},
        'band_powers_per_channel': band_powers,
        
        # Engagement ratio (beta/theta)
        'engagement_ratio': np.mean(attention_indices),
        
        # Alertness (theta power - inverse)
        'alertness': 1.0 / (np.mean(band_powers['theta']) + 1e-10)
    }
    
    print(f"  ✓ Attention index: {biomarkers['attention_index']:.3f}")
    print(f"  ✓ Motor control index: {biomarkers['motor_control_index']:.3f}")
    print(f"  ✓ Mu suppression: {biomarkers['mu_suppression']:.3f}")
    
    return biomarkers


# =============================================================================
# STEP 6: COMPARISON
# =============================================================================

def compare_periods(baseline_bio, task_bio):
    """Compare baseline vs task biomarkers"""
    print("\n" + "="*70)
    print("COMPARING BASELINE vs TASK")
    print("="*70)
    
    comparison = {}
    
    # Attention change
    att_change = ((task_bio['attention_index'] - baseline_bio['attention_index']) / 
                  (baseline_bio['attention_index'] + 1e-10)) * 100
    
    # Motor control change
    motor_change = ((task_bio['motor_control_index'] - baseline_bio['motor_control_index']) / 
                    (baseline_bio['motor_control_index'] + 1e-10)) * 100
    
    # Mu suppression change (should increase during motor tasks)
    mu_change = ((task_bio['mu_suppression'] - baseline_bio['mu_suppression']) / 
                 (baseline_bio['mu_suppression'] + 1e-10)) * 100
    
    comparison = {
        'attention': {
            'baseline': baseline_bio['attention_index'],
            'task': task_bio['attention_index'],
            'change_%': att_change
        },
        'motor_control': {
            'baseline': baseline_bio['motor_control_index'],
            'task': task_bio['motor_control_index'],
            'change_%': motor_change
        },
        'mu_suppression': {
            'baseline': baseline_bio['mu_suppression'],
            'task': task_bio['mu_suppression'],
            'change_%': mu_change
        },
        'engagement': {
            'baseline': baseline_bio['engagement_ratio'],
            'task': task_bio['engagement_ratio'],
            'change_%': att_change  # Same as attention
        }
    }
    
    print(f"  Attention change: {att_change:+.1f}%")
    print(f"  Motor control change: {motor_change:+.1f}%")
    print(f"  Mu suppression change: {mu_change:+.1f}%")
    
    return comparison


# =============================================================================
# STEP 7: FEEDBACK GENERATION
# =============================================================================

def generate_feedback(baseline_bio, task_bio, comparison, task_name):
    """Generate comprehensive feedback report"""
    
    fb = []
    fb.append("\n" + "="*70)
    fb.append("REHABILITATION FEEDBACK REPORT")
    fb.append("="*70)
    fb.append(f"Task: {task_name}")
    fb.append("="*70)
    
    # ATTENTION/FOCUS ANALYSIS
    fb.append("\n🧠 ATTENTION & FOCUS")
    fb.append("-"*70)
    
    att_change = comparison['attention']['change_%']
    att_task = comparison['attention']['task']
    
    # Interpret attention level
    if att_task > 2.0:
        att_level = "EXCELLENT"
        att_emoji = "✓✓"
    elif att_task > 1.5:
        att_level = "GOOD"
        att_emoji = "✓"
    elif att_task > 1.0:
        att_level = "MODERATE"
        att_emoji = "−"
    else:
        att_level = "LOW"
        att_emoji = "⚠"
    
    fb.append(f"{att_emoji} Attention Level: {att_level} ({att_task:.2f})")
    
    # Interpret attention change
    if att_change > 15:
        fb.append(f"✓ Focus IMPROVED significantly during task (+{att_change:.1f}%)")
        fb.append("  → Good mental engagement with the exercise")
    elif att_change > 5:
        fb.append(f"✓ Focus IMPROVED during task (+{att_change:.1f}%)")
        fb.append("  → Maintained good attention throughout")
    elif att_change > -5:
        fb.append(f"− Focus remained STABLE ({att_change:+.1f}%)")
        fb.append("  → Consistent attention level")
    elif att_change > -15:
        fb.append(f"⚠ Focus DECREASED slightly during task ({att_change:.1f}%)")
        fb.append("  → Consider: Task may be too long or too difficult")
    else:
        fb.append(f"⚠⚠ Focus DECREASED significantly during task ({att_change:.1f}%)")
        fb.append("  → Recommendation: Shorten session or add breaks")
        fb.append("  → Check for fatigue or distraction")
    
    # MOTOR CONTROL ANALYSIS
    fb.append("\n🎯 MOTOR CONTROL & ENGAGEMENT")
    fb.append("-"*70)
    
    motor_change = comparison['motor_control']['change_%']
    motor_task = comparison['motor_control']['task']
    
    # Interpret motor control level
    if motor_task > 2.0:
        motor_level = "HIGH"
        motor_emoji = "✓✓"
    elif motor_task > 1.5:
        motor_level = "GOOD"
        motor_emoji = "✓"
    elif motor_task > 1.0:
        motor_level = "MODERATE"
        motor_emoji = "−"
    else:
        motor_level = "LOW"
        motor_emoji = "⚠"
    
    fb.append(f"{motor_emoji} Motor Control: {motor_level} ({motor_task:.2f})")
    
    # Interpret motor control change
    if motor_change > 20:
        fb.append(f"✓✓ Motor engagement INCREASED strongly (+{motor_change:.1f}%)")
        fb.append("  → Excellent motor system activation")
        fb.append("  → Brain is actively planning and executing movements")
    elif motor_change > 10:
        fb.append(f"✓ Motor engagement INCREASED (+{motor_change:.1f}%)")
        fb.append("  → Good motor control during task")
    elif motor_change > -10:
        fb.append(f"− Motor engagement STABLE ({motor_change:+.1f}%)")
        fb.append("  → Consistent motor control")
    else:
        fb.append(f"⚠ Motor engagement DECREASED ({motor_change:.1f}%)")
        fb.append("  → May indicate fatigue or reduced effort")
        fb.append("  → Consider: Simplify task or provide more feedback")
    
    # MU RHYTHM SUPPRESSION (Motor Preparation)
    fb.append("\n⚡ MOTOR PREPARATION (Mu Rhythm)")
    fb.append("-"*70)
    
    mu_change = comparison['mu_suppression']['change_%']
    
    if mu_change > 15:
        fb.append(f"✓✓ Strong mu suppression during task (+{mu_change:.1f}%)")
        fb.append("  → Excellent motor preparation and execution")
        fb.append("  → Brain is effectively planning movements")
    elif mu_change > 5:
        fb.append(f"✓ Good mu suppression (+{mu_change:.1f}%)")
        fb.append("  → Appropriate motor preparation")
    elif mu_change > -5:
        fb.append(f"− Minimal mu suppression change ({mu_change:+.1f}%)")
        fb.append("  → Task may be more cognitive than motor")
    else:
        fb.append(f"⚠ Reduced mu suppression ({mu_change:.1f}%)")
        fb.append("  → Limited motor system engagement")
        fb.append("  → Consider: More physically engaging exercises")
    
    # OVERALL ASSESSMENT
    fb.append("\n📊 OVERALL ASSESSMENT")
    fb.append("-"*70)
    
    # Calculate overall score
    scores = []
    
    # Attention score
    if att_change > 5:
        scores.append(2)
    elif att_change > -5:
        scores.append(1)
    else:
        scores.append(0)
    
    # Motor score
    if motor_change > 10:
        scores.append(2)
    elif motor_change > -10:
        scores.append(1)
    else:
        scores.append(0)
    
    # Mu score
    if mu_change > 5:
        scores.append(2)
    elif mu_change > -5:
        scores.append(1)
    else:
        scores.append(0)
    
    overall_score = sum(scores)
    
    if overall_score >= 5:
        fb.append("✓✓ EXCELLENT SESSION")
        fb.append("  • Strong attention and focus maintained")
        fb.append("  • High motor control engagement")
        fb.append("  • Effective motor preparation")
        fb.append("  → Continue current rehabilitation protocol")
    elif overall_score >= 3:
        fb.append("✓ GOOD SESSION")
        fb.append("  • Adequate attention and motor control")
        fb.append("  • Task is appropriate for current level")
        fb.append("  → Continue with gradual progression")
    else:
        fb.append("⚠ NEEDS ADJUSTMENT")
        fb.append("  • Consider modifying task parameters:")
        
        if att_change < -5:
            fb.append("    - Shorten session duration")
            fb.append("    - Add rest breaks")
        
        if motor_change < -10:
            fb.append("    - Simplify motor requirements")
            fb.append("    - Provide more feedback/guidance")
        
        if mu_change < -5:
            fb.append("    - Increase physical engagement")
            fb.append("    - Add more movement components")
    
    # RECOMMENDATIONS
    fb.append("\n💡 RECOMMENDATIONS")
    fb.append("-"*70)
    
    # Personalized recommendations based on results
    recs = []
    
    if att_task < 1.5:
        recs.append("• Attention Training: Practice focus exercises before motor tasks")
    
    if motor_task < 1.5:
        recs.append("• Motor Engagement: Start with simpler movements to build confidence")
    
    if att_change < -10:
        recs.append("• Session Length: Reduce duration to 15-20 minutes initially")
        recs.append("• Break Strategy: Add 2-3 minute breaks every 10 minutes")
    
    if motor_change > 20 and att_change > 10:
        recs.append("• Ready for Progression: Consider increasing task difficulty")
    
    if mu_change < 0:
        recs.append("• Motor Activation: Add more physically demanding movements")
        recs.append("• Imagery Practice: Include motor imagery exercises")
    
    # Add context-based recommendations
    theta_task = task_bio['band_powers']['theta']
    theta_baseline = baseline_bio['band_powers']['theta']
    
    if theta_task > theta_baseline * 1.2:
        recs.append("• Fatigue Detected: Ensure adequate rest between sessions")
        recs.append("• Sleep Quality: Monitor and optimize sleep patterns")
    
    if len(recs) == 0:
        recs.append("• Excellent progress - maintain current program")
        recs.append("• Track metrics over time to monitor improvement")
    
    for rec in recs:
        fb.append(rec)
    
    # PROGRESS TRACKING
    fb.append("\n📈 FOR NEXT SESSION")
    fb.append("-"*70)
    fb.append("Track these metrics to monitor progress:")
    fb.append(f"  • Attention Index: {task_bio['attention_index']:.2f}")
    fb.append(f"  • Motor Control Index: {task_bio['motor_control_index']:.2f}")
    fb.append(f"  • Mu Suppression: {task_bio['mu_suppression']:.2f}")
    fb.append("\nGoals for improvement:")
    
    if att_task < 2.0:
        target = att_task * 1.15
        fb.append(f"  • Increase attention to {target:.2f} (+15%)")
    
    if motor_task < 2.0:
        target = motor_task * 1.15
        fb.append(f"  • Increase motor control to {target:.2f} (+15%)")
    
    fb.append("\n" + "="*70)
    
    return "\n".join(fb)


# =============================================================================
# STEP 8: VISUALIZATION
# =============================================================================

def create_visualization(baseline_bio, task_bio, comparison, task_name):
    """Create comprehensive EEG analysis visualization"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    fig.suptitle(f'EEG Analysis: {task_name}', fontsize=16, fontweight='bold', y=0.98)
    
    # Plot 1: Attention Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    periods = ['Baseline', 'Task']
    att_vals = [comparison['attention']['baseline'], comparison['attention']['task']]
    bars = ax1.bar(periods, att_vals, color=['skyblue', 'coral'], alpha=0.7, width=0.6)
    ax1.set_ylabel('Attention Index', fontsize=11)
    ax1.set_title('Attention / Focus Level', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, att_vals):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add interpretation
    att_level = "High" if att_vals[1] > 2.0 else "Good" if att_vals[1] > 1.5 else "Moderate"
    ax1.text(0.5, 0.95, f'Task Level: {att_level}', transform=ax1.transAxes,
            ha='center', va='top', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Plot 2: Motor Control Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    motor_vals = [comparison['motor_control']['baseline'], comparison['motor_control']['task']]
    bars = ax2.bar(periods, motor_vals, color=['lightgreen', 'darkgreen'], alpha=0.7, width=0.6)
    ax2.set_ylabel('Motor Control Index', fontsize=11)
    ax2.set_title('Motor Control Engagement', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, motor_vals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    motor_level = "High" if motor_vals[1] > 2.0 else "Good" if motor_vals[1] > 1.5 else "Moderate"
    ax2.text(0.5, 0.95, f'Task Level: {motor_level}', transform=ax2.transAxes,
            ha='center', va='top', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # Plot 3: Change Percentages
    ax3 = fig.add_subplot(gs[0, 2])
    metrics = ['Attention', 'Motor\nControl', 'Mu\nSuppression']
    changes = [
        comparison['attention']['change_%'],
        comparison['motor_control']['change_%'],
        comparison['mu_suppression']['change_%']
    ]
    colors = ['green' if c > 5 else 'red' if c < -5 else 'gray' for c in changes]
    bars = ax3.barh(metrics, changes, color=colors, alpha=0.7)
    ax3.axvline(0, color='black', linewidth=1.5)
    ax3.axvline(5, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
    ax3.axvline(-5, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    ax3.set_xlabel('Change (%)', fontsize=11)
    ax3.set_title('Baseline → Task Changes', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars, changes):
        width = bar.get_width()
        ax3.text(width + (2 if width > 0 else -2), bar.get_y() + bar.get_height()/2.,
                f'{val:+.1f}%', ha='left' if val > 0 else 'right',
                va='center', fontweight='bold', fontsize=9)
    
    # Plot 4: Band Power Distribution (Baseline)
    ax4 = fig.add_subplot(gs[1, 0])
    bands = list(baseline_bio['band_powers'].keys())
    baseline_powers = [baseline_bio['band_powers'][b] for b in bands]
    colors_bands = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    ax4.bar(bands, baseline_powers, color=colors_bands, alpha=0.7)
    ax4.set_ylabel('Power (μV²)', fontsize=10)
    ax4.set_title('Baseline - Frequency Bands', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.tick_params(axis='x', labelsize=9)
    
    # Plot 5: Band Power Distribution (Task)
    ax5 = fig.add_subplot(gs[1, 1])
    task_powers = [task_bio['band_powers'][b] for b in bands]
    ax5.bar(bands, task_powers, color=colors_bands, alpha=0.7)
    ax5.set_ylabel('Power (μV²)', fontsize=10)
    ax5.set_title('Task - Frequency Bands', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.tick_params(axis='x', labelsize=9)
    
    # Plot 6: Band Power Changes
    ax6 = fig.add_subplot(gs[1, 2])
    power_changes = [(task_powers[i] - baseline_powers[i]) / (baseline_powers[i] + 1e-10) * 100 
                     for i in range(len(bands))]
    colors_change = ['green' if c > 0 else 'red' for c in power_changes]
    bars = ax6.barh(bands, power_changes, color=colors_change, alpha=0.7)
    ax6.axvline(0, color='black', linewidth=1)
    ax6.set_xlabel('Change (%)', fontsize=10)
    ax6.set_title('Band Power Changes', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    ax6.tick_params(axis='y', labelsize=9)
    
    for bar, val in zip(bars, power_changes):
        width = bar.get_width()
        ax6.text(width + (2 if width > 0 else -2), bar.get_y() + bar.get_height()/2.,
                f'{val:+.0f}%', ha='left' if val > 0 else 'right',
                va='center', fontsize=8)
    
    # Plot 7: Per-Channel Attention
    ax7 = fig.add_subplot(gs[2, 0])
    channels = [f'Ch{i+1}' for i in range(len(task_bio['attention_per_channel']))]
    att_per_ch = task_bio['attention_per_channel']
    ax7.barh(channels, att_per_ch, color='steelblue', alpha=0.7)
    ax7.axvline(task_bio['attention_index'], color='red', 
               linestyle='--', linewidth=2, label='Average')
    ax7.set_xlabel('Attention Index', fontsize=10)
    ax7.set_title('Task - Attention by Channel', fontsize=11, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3, axis='x')
    ax7.tick_params(axis='y', labelsize=9)
    
    # Plot 8: Per-Channel Motor Control
    ax8 = fig.add_subplot(gs[2, 1])
    motor_per_ch = task_bio['motor_per_channel']
    ax8.barh(channels, motor_per_ch, color='darkgreen', alpha=0.7)
    ax8.axvline(task_bio['motor_control_index'], color='red', 
               linestyle='--', linewidth=2, label='Average')
    ax8.set_xlabel('Motor Control Index', fontsize=10)
    ax8.set_title('Task - Motor Control by Channel', fontsize=11, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3, axis='x')
    ax8.tick_params(axis='y', labelsize=9)
    
    # Plot 9: Summary & Recommendations
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Generate summary text
    att_change = comparison['attention']['change_%']
    motor_change = comparison['motor_control']['change_%']
    mu_change = comparison['mu_suppression']['change_%']
    
    summary = "SUMMARY\n" + "="*30 + "\n\n"
    
    # Overall status
    if att_change > 5 and motor_change > 10:
        summary += "✓✓ EXCELLENT\n"
        summary += "Strong engagement\n\n"
    elif att_change > -5 and motor_change > -10:
        summary += "✓ GOOD\n"
        summary += "Adequate performance\n\n"
    else:
        summary += "⚠ NEEDS ATTENTION\n"
        summary += "Consider adjustments\n\n"
    
    summary += "Key Metrics:\n"
    summary += f"• Attention: {comparison['attention']['task']:.2f}\n"
    summary += f"  ({att_change:+.1f}% change)\n\n"
    summary += f"• Motor Control: {comparison['motor_control']['task']:.2f}\n"
    summary += f"  ({motor_change:+.1f}% change)\n\n"
    summary += f"• Mu Suppression:\n"
    summary += f"  ({mu_change:+.1f}% change)\n\n"
    
    summary += "-"*30 + "\n"
    summary += "Quick Tips:\n"
    
    if att_change < -10:
        summary += "• Add breaks\n"
    if motor_change < -10:
        summary += "• Simplify task\n"
    if mu_change < 0:
        summary += "• More movement\n"
    if att_change > 10 and motor_change > 10:
        summary += "• Ready to progress\n"
    
    ax9.text(0.05, 0.95, summary, transform=ax9.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_analysis(data_file, task_name, baseline_idx, task_idx, fs):
    """Complete EEG analysis pipeline"""
    
    print("\n" + "="*70)
    print("EEG MOTOR CONTROL REHABILITATION ANALYSIS")
    print("="*70)
    print(f"Task: {task_name}")
    print(f"Sampling Rate: {fs} Hz")
    print("="*70)
    
    # Step 1: Load
    df = load_data(data_file)
    if df is None:
        return None
    
    # Step 2: Find markers
    markers = find_markers(df)
    
    # Step 3: Segment
    segments = segment_data(df, markers, baseline_idx, task_idx)
    if 'baseline' not in segments or 'task' not in segments:
        print("\n✗ ERROR: Could not segment data!")
        return None
    
    # Analyze both periods
    results = {}
    
    for period in ['baseline', 'task']:
        print("\n" + "="*70)
        print(f"ANALYZING {period.upper()} PERIOD")
        print("="*70)
        
        # Step 4: Extract and preprocess EEG
        print(f"Extracting EEG data...")
        eeg_raw = extract_eeg(segments[period])
        print(f"  ✓ EEG shape: {eeg_raw.shape}")
        
        print(f"Preprocessing EEG...")
        eeg_clean = preprocess_eeg(eeg_raw, fs)
        print(f"  ✓ Filtered (0.5-50 Hz)")
        
        # Step 5: Extract biomarkers
        biomarkers = extract_biomarkers(eeg_clean, fs)
        
        results[period] = biomarkers
    
    # Step 6: Compare
    comparison = compare_periods(results['baseline'], results['task'])
    
    # Step 7: Generate feedback
    feedback = generate_feedback(results['baseline'], results['task'], 
                                 comparison, task_name)
    print(feedback)
    
    # Save feedback to file
    feedback_filename = 'eeg_feedback_report.txt'
    try:
        with open(feedback_filename, 'w', encoding='utf-8') as f:
            f.write(feedback)
        print(f"\n✓ Feedback saved to: {feedback_filename}")
    except Exception as e:
        print(f"\n⚠ Could not save feedback file: {e}")
    
    # Step 8: Visualize
    if SAVE_FIGURES or SHOW_PLOTS:
        print("\n" + "="*70)
        print("GENERATING VISUALIZATION")
        print("="*70)
        
        fig = create_visualization(results['baseline'], results['task'], 
                                   comparison, task_name)
        
        if SAVE_FIGURES:
            filename = 'eeg_analysis_summary.png'
            fig.savefig(filename, dpi=OUTPUT_DPI, bbox_inches='tight')
            print(f"  ✓ Saved: {filename}")
            
            # Automatically open the saved image
            try:
                import os
                import platform
                
                abs_path = os.path.abspath(filename)
                
                if platform.system() == 'Windows':
                    os.startfile(abs_path)
                    print(f"  ✓ Opening image automatically...")
                elif platform.system() == 'Darwin':  # macOS
                    os.system(f'open "{abs_path}"')
                    print(f"  ✓ Opening image automatically...")
                else:  # Linux
                    os.system(f'xdg-open "{abs_path}"')
                    print(f"  ✓ Opening image automatically...")
            except Exception as e:
                print(f"  ℹ Could not auto-open image (you can open it manually): {e}")
        
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE!")
    print("="*70)
    
    return {
        'baseline': results['baseline'],
        'task': results['task'],
        'comparison': comparison,
        'feedback': feedback
    }


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    
    if DATA_FILE == "your_eeg_recording.txt":
        print("\n" + "="*70)
        print("⚠ CONFIGURATION NEEDED")
        print("="*70)
        print("\nPlease edit the CONFIGURATION section (lines 30-45):")
        print("1. Set DATA_FILE to your actual file path")
        print("2. Set TASK_NAME to describe your motor task")
        print("3. Run this script again\n")
        print("Example:")
        print('  DATA_FILE = "C:/Data/reaching_task_session1.txt"')
        print('  TASK_NAME = "Reaching Task"')
        print("="*70)
    else:
        results = run_analysis(
            data_file=DATA_FILE,
            task_name=TASK_NAME,
            baseline_idx=BASELINE_MARKERS,
            task_idx=TASK_MARKERS,
            fs=SAMPLING_RATE
        )
        
        if results:
            print("\n✓ All biomarkers extracted!")
            print("✓ Feedback generated!")
            if SAVE_FIGURES:
                print("✓ Visualization saved!")