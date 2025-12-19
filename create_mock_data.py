
import os
import shutil
import pandas as pd
import numpy as np
import nibabel as nib
import traceback

def create_mock_data():
    try:
        # Define paths
        data_dir = "./Data"
        tsv_path = "./datasets/files/Train_diagnosis_ADNI.tsv"
        val_tsv_path = "./datasets/files/Val_diagnosis_ADNI.tsv"
        
        # Read a few subjects from the TSV
        df = pd.read_csv(tsv_path, sep='\t')
        
        # Process first 10 entries for mock data
        subset = df.head(10)
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        print(f"Creating mock data in {data_dir} for {len(subset)} subjects...")

        for index, row in subset.iterrows():
            subject_id = row['participant_id']
            session_id = row['session_id']
            
            # Construct path: Data/<subject_id>/<session_id>/t1/spm/segmentation/normalized_space
            # Note: The code says 't1/spm/segmentation/normalized_space' relative to session dir
            
            target_dir = os.path.join(data_dir, subject_id, session_id, 't1', 'spm', 'segmentation', 'normalized_space')
            
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            # Create a dummy NIfTI file
            # Code expects: if 'Space_T1w' in seg_name:
            file_name = "Space_T1w_mock.nii"
            file_path = os.path.join(target_dir, file_name)
            
            # Create a small random 3D array
            # Code crops to 96x96x96, so let's make it slightly larger or exact
            # adni_3d.py: image = self.randomCrop(image,96,96,96)
            data = np.random.rand(100, 100, 100).astype(np.float32)
            
            # Save as NIfTI
            img = nib.Nifti1Image(data, np.eye(4))
            nib.save(img, file_path)
            
            print(f"Created {file_path}")

        # Also create for Validation set (first 5)
        df_val = pd.read_csv(val_tsv_path, sep='\t')
        subset_val = df_val.head(5)
        print(f"Creating mock data for validation set ({len(subset_val)} subjects)...")
        
        for index, row in subset_val.iterrows():
            subject_id = row['participant_id']
            session_id = row['session_id']
            target_dir = os.path.join(data_dir, subject_id, session_id, 't1', 'spm', 'segmentation', 'normalized_space')
             
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            file_name = "Space_T1w_mock.nii"
            file_path = os.path.join(target_dir, file_name)
            
            if not os.path.exists(file_path): # Don't overwrite if already there from train
                data = np.random.rand(100, 100, 100).astype(np.float32)
                img = nib.Nifti1Image(data, np.eye(4))
                nib.save(img, file_path)
                print(f"Created {file_path}")

        print("Mock data creation completed.")

    except Exception as e:
        print("Error creating mock data:")
        print(traceback.format_exc())

if __name__ == "__main__":
    create_mock_data()
