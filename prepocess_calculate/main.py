
import subprocess

def run_streamlit():
    try:
        # Run the streamlit command
        subprocess.run(["streamlit", "run", "streamlit_app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")

if __name__ == "__main__":
    run_streamlit()
