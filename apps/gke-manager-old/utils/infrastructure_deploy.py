import streamlit as st
import subprocess


def configure_infrastructure(select_bash_script):
    output_text = ""
    # Run the script using subprocess.Popen
    with st.spinner(f":red[**Running {select_bash_script.split('/')[-1]} script, Please wait...**]"):
        text_placeholder = st.empty()
        try:
            process = subprocess.Popen(
                [select_bash_script],  # Command to run
                stdout=subprocess.PIPE,  # Capture stdout
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                text=True  # Decode output as text
            )

            # Stream the output in real-time
            while True:
                line = process.stdout.readline()  # Read one line at a time
                if not line and process.poll() is not None:  # Exit loop when process ends
                    break
                # output_text += line  # Append the new line to the buffer
                output_text += f"{'-' * 50}\n{line.strip()}\n"
                text_placeholder.text_area("", value=output_text, height=400)  # Update the placeholder

            # Check if the process exited successfully
            if process.returncode == 0:
                st.success("Script executed successfully!")
            else:
                st.error(f"Script exited with return code {process.returncode}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
