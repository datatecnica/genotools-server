import requests

def send_email_via_courier(api_key, template_id, recipient_email, name, message):
    """
    Sends an email using the Courier API.

    Parameters:
        api_key (str): Your Courier API key.
        template_id (str): The ID of the email template created in Courier.
        recipient_email (str): The recipient's email address.
        name (str): The name of the recipient (used in the email template).
        message (str): The custom message to include in the email.

    Returns:
        dict: Response from the Courier API.
    """
    # Define the API endpoint
    url = "https://api.courier.com/send"

    # Define the headers with the API key
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Define the payload with template data and recipient information
    payload = {
        "message": {
            "template": template_id,
            "to": {
                "email": recipient_email
            },
            "data": {
                "name": name,
                "message": message
            }
        }
    }

    # Send the POST request to the Courier API
    response = requests.post(url, json=payload, headers=headers)

    # Check the response status
    if response.status_code == 200:
        print("Email sent successfully!")
    else:
        print(f"Failed to send email. Status code: {response.status_code}")
        print(f"Response: {response.text}")

    return response.json()

# Example usage
if __name__ == "__main__":
    # Replace these values with your own
    api_key = "your_courier_api_key"
    template_id = "your_template_id"
    recipient_email = "recipient@example.com"
    name = "John Doe"
    message = "This is a test message sent using the Courier API."

    # Send the email
    response = send_email_via_courier(api_key, template_id, recipient_email, name, message)
    print("API Response:", response)