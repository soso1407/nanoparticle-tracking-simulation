"""
Results Migration Tool

This script migrates existing results to the new folder structure.
Original files remain in the results folder, but copies are made in 
the iteration_1 directory for consistent organization.
"""

import os
import shutil
import glob

def migrate_results():
    """
    Migrate existing result files to the iteration_1 folder.
    """
    # Create iteration_1 directory if it doesn't exist
    iteration1_dir = os.path.join('results', 'iteration_1')
    os.makedirs(iteration1_dir, exist_ok=True)
    
    # Get all files in the root results directory
    results_dir = 'results'
    files_to_migrate = []
    
    for file in os.listdir(results_dir):
        # Skip iteration folders
        if file.startswith('iteration_'):
            continue
        
        # Get full path
        full_path = os.path.join(results_dir, file)
        
        # Only migrate files, not directories
        if os.path.isfile(full_path):
            files_to_migrate.append(full_path)
    
    # Copy each file to iteration_1
    for file_path in files_to_migrate:
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(iteration1_dir, file_name)
        
        # Copy the file
        shutil.copy2(file_path, dest_path)
        print(f"Copied {file_name} to {iteration1_dir}")
    
    print(f"Migration complete. {len(files_to_migrate)} files migrated to {iteration1_dir}")

if __name__ == "__main__":
    migrate_results() 