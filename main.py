import subprocess
import sys

def run_script(script_name):
    result = subprocess.run([sys.executable, script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(f"Running {script_name}:")
    print(result.stdout)

if __name__ == "__main__":
    scripts = ["image_scraper.py", "ringworm_classifier_V2.py", "disperser.py", "dupe_delete.py"]

    for script in scripts:
        run_script(script)
