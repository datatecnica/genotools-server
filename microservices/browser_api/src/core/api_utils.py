import json
import requests

def format_error_response(response_text):
    try:
        # Parse the JSON response
        response_data = json.loads(response_text)
        
        # Extract the error detail
        if 'detail' in response_data:
            error_detail = response_data['detail']
            
            # Split the error message from the traceback
            if '\n\n\nTraceback:' in error_detail:
                error_message, traceback = error_detail.split('\n\n\nTraceback:', 1)
            else:
                error_message = error_detail
                traceback = ""
            
            # Display the error message
            print("Error Message:")
            print(f"{error_message.strip()}")
            
            # Print traceback if available
            if traceback:
                print("\nTraceback:")
                print(traceback)
        else:
            # Pretty-print the entire JSON
            formatted_json = json.dumps(response_data, indent=2)
            print(formatted_json)
    except json.JSONDecodeError:
        # If not valid JSON, just print the text
        print(response_text)


def post_to_api(url, payload):
    # Make the API request
    response = requests.post(url, json=payload)
    
    # Handle the response
    if response.status_code == 200:
        data = response.json()
        print(f"Status code: {response.status_code} - Success!")
        
        # Display file paths if available
        if 'outputs' in data:
            print("Output Files:")
            for path in data['outputs']:
                print(f"{path}")
        print()
    
    else:
        # Print status code for error cases too
        print(f"Status code: {response.status_code}")
        
        # Format error response
        try:
            format_error_response(response.text)
        except:
            print(f"Error: {response.text}")
        print()