import streamlit as st
import subprocess
from streamlit_ace import st_ace
import io
import sys

def api_call():
    # python_code =  st_ace("Please adjust your parameters in the code here:", height=200, value="""
    python_code = st.text_area(":red[**Please adjust your parameters in the code here and Indent Properly:**]", height=200)  

    # Button to execute the Python code
    if st.button("Submit Workload") and python_code:
        if python_code.strip() == "":
            st.error("Please enter some Python code to execute.")
        else:
            # Redirect stdout to capture the output
            old_stdout = sys.stdout
            new_stdout = io.StringIO()
            sys.stdout = new_stdout

            try:
                # Execute the Python code
                exec(python_code)
                output = new_stdout.getvalue().strip()  # Get the captured output

                # Display the output in a text area
                st.subheader("Output")
                if output:
                    st.text_area("Result", value=output, height=200)
                else:
                    st.warning("The code executed successfully, but no output was generated.")

            except Exception as e:
                # Capture and display any errors
                st.subheader("Error")
                st.text_area("Error Message", value=str(e), height=200)

            finally:
                # Restore the original stdout
                sys.stdout = old_stdout
