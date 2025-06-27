
import numpy as np
import pandas as pd
import sys
sys.path.append('.')

from utils.kaggle_submission import KaggleSubmissionGenerator

def test_submission_pipeline():
    """Test the submission pipeline with sample data"""
    
    print("=== Testing Submission Pipeline ===\n")
    
    # Initialize
    kaggle_gen = KaggleSubmissionGenerator()
    
    # Create sample test files (using your actual test files)
    test_files = ["000030dca2", "000031dca3", "000032dca4", "000033dca5", "000034dca6"]
    print(f"Testing with {len(test_files)} files: {test_files}")
    
    # Generate sample predictions
    predictions = {}
    for file_id in test_files:
        # Create 70x70 velocity map with realistic seismic velocities
        velocity_map = np.random.normal(3000, 500, (70, 70))
        velocity_map = np.clip(velocity_map, 1500, 6000)
        predictions[file_id] = velocity_map
    
    print(f"Generated predictions for {len(predictions)} files")
    
    # Create submission
    try:
        submission_df = kaggle_gen.create_submission_from_predictions(
            predictions=predictions,
            output_path="test_submission.csv"
        )
        
        print(f"\n[SUCCESS] Submission created!")
        print(f"Shape: {submission_df.shape}")
        print(f"Expected rows: {len(test_files) * 70} (5 files × 70 rows each)")
        print(f"Expected columns: 36 (1 ID + 35 velocity columns)")
        
        # Show sample rows
        print("\nFirst 3 rows:")
        print(submission_df.head(3))
        
        # Validate
        validation = kaggle_gen._validate_submission(submission_df)
        print(f"\nValidation results:")
        print(f"Valid: {validation['is_valid']}")
        print(f"MAE ready: {validation['mae_ready']}")
        
        if validation['errors']:
            print(f"Errors: {validation['errors']}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Submission failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_submission_pipeline()
    if success:
        print("\n[SUCCESS] Pipeline test completed successfully!")
    else:
        print("\n[ERROR] Pipeline test failed!")
