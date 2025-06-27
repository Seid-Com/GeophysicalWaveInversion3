
#!/usr/bin/env python3
"""
Main script to create Kaggle submission for FWI competition
"""

import numpy as np
import torch
from pathlib import Path
import sys
sys.path.append('.')

from utils.kaggle_submission import KaggleSubmissionGenerator
from models.physics_guided_network import PhysicsGuidedFWI
from utils.preprocessor import SeismicPreprocessor

def create_kaggle_submission():
    """Create submission file for Kaggle competition"""
    
    print("=== Creating Kaggle FWI Submission ===\n")
    
    # Initialize submission generator
    kaggle_gen = KaggleSubmissionGenerator()
    
    # Load test file IDs
    test_dir = "sample_data/sample_data/test"
    test_file_ids = kaggle_gen.load_test_file_ids(test_dir)
    
    print(f"Found {len(test_file_ids)} test files: {test_file_ids}\n")
    
    # Initialize your trained model
    print("Loading physics-guided FWI model...")
    model = PhysicsGuidedFWI(
        input_channels=1,
        output_channels=1,
        encoder_channels=[64, 128, 256],
        decoder_channels=[256, 128, 64]
    )
    
    # Load trained weights if available
    # model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # Generate predictions for each test file
    print("Generating velocity predictions...")
    predictions = {}
    preprocessor = SeismicPreprocessor()
    
    for i, file_id in enumerate(test_file_ids):
        print(f"Processing {file_id} ({i+1}/{len(test_file_ids)})")
        
        # Load test data
        test_file = Path(test_dir) / f"{file_id}.npy"
        
        if test_file.exists():
            seismic_data = np.load(test_file)
        else:
            # Create dummy data if file doesn't exist
            seismic_data = np.random.normal(0, 1, (1, 50, 70))
        
        # Preprocess data
        if len(seismic_data.shape) == 3:
            seismic_data = seismic_data[np.newaxis, :]
        
        seismic_tensor = torch.FloatTensor(seismic_data)
        
        # Generate prediction
        with torch.no_grad():
            prediction = model(seismic_tensor)
            prediction_np = prediction.cpu().numpy()
        
        # Store prediction (should be 70x70 velocity map)
        predictions[file_id] = prediction_np
    
    print(f"\nGenerated predictions for {len(predictions)} files")
    
    # Create Kaggle submission file
    print("Creating Kaggle submission file...")
    submission_df = kaggle_gen.create_submission_from_predictions(
        predictions=predictions,
        output_path="kaggle_submission.csv"
    )
    
    # Validate submission
    validation = kaggle_gen._validate_submission(submission_df)
    print(f"\nSubmission validation:")
    print(f"✅ Valid: {validation['is_valid']}")
    print(f"✅ Total rows: {validation['total_rows']:,}")
    print(f"✅ Total files: {validation['total_files']}")
    
    if validation['errors']:
        print(f"❌ Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"⚠️  Warnings: {validation['warnings']}")
    
    # Preview submission
    print("\n=== Submission Preview ===")
    print(submission_df.head())
    
    print(f"\n🎯 Submission ready!")
    print(f"📁 File: kaggle_submission.csv")
    print(f"📊 Format: {submission_df.shape[0]} rows × {submission_df.shape[1]} columns")
    print(f"🔗 Upload to: https://www.kaggle.com/competitions/waveform-inversion/submissions")
    
    return submission_df

def create_sample_submission():
    """Create a sample submission with default values"""
    
    print("=== Creating Sample Submission ===\n")
    
    kaggle_gen = KaggleSubmissionGenerator()
    
    # Load test file IDs
    test_dir = "sample_data/sample_data/test"
    test_file_ids = kaggle_gen.load_test_file_ids(test_dir)
    
    # Create sample submission with default velocity (3000 m/s)
    sample_df = kaggle_gen.create_sample_submission(
        test_file_ids=test_file_ids,
        output_path="sample_kaggle_submission.csv"
    )
    
    print("✅ Sample submission created!")
    print(f"📁 File: sample_kaggle_submission.csv")
    print(f"📊 Shape: {sample_df.shape}")
    
    return sample_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Kaggle FWI submission")
    parser.add_argument("--sample", action="store_true", 
                       help="Create sample submission with default values")
    
    args = parser.parse_args()
    
    if args.sample:
        create_sample_submission()
    else:
        create_kaggle_submission()
