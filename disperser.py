import os
import shutil

def move_files(src_folder, dest_folder):
    for filename in os.listdir(src_folder):
        src = os.path.join(src_folder, filename)
        dest = os.path.join(dest_folder, filename)
        shutil.move(src, dest.replace("\\", "/"))

if __name__ == "__main__":
    tested_folder = r"C:\Users\Danie\OneDrive\Desktop\ringworm_classifier_V2\tested"
    ringworm_folder = r"C:\Users\Danie\OneDrive\Desktop\ringworm_classifier_V2\data\Ringworm"

    untested_folder = r"C:\Users\Danie\OneDrive\Desktop\ringworm_classifier_V2\untested"
    not_ringworm_folder = r"C:\Users\Danie\OneDrive\Desktop\ringworm_classifier_V2\data\NotRingworm"

    move_files(tested_folder, ringworm_folder)
    move_files(untested_folder, not_ringworm_folder)
