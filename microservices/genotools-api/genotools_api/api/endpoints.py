from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader, APIKey
from typing import List
import logging
import os
from google.cloud import secretmanager
from dotenv import load_dotenv
from genotools_api.models.models import GenoToolsParams
from genotools_api.utils.utils import download_from_gcs, construct_command, execute_genotools, upload_to_gcs
from time import sleep

#This section sends email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from datetime import datetime

# from utils.email import send_email_via_courier

# def noreply_email():
#     api_key = "your_courier_api_key"
#     template_id = "your_template_id"
#     recipient_email = "recipient@example.com"
#     name = "John Doe"
#     message = "This is a test message sent using the Courier API."

#     # Send the email
#     response = send_email_via_courier(api_key, template_id, recipient_email, name, message)


subject_submitted = "Job Submission Confirmation (gtserver: genotools-api)"
subject_completed = "Job Completion Confirmation (gtserver: genotools-api)"
body_submitted = 'You genotools-api job has been submitted. You will receive an eamil upon job completion as well.'
body_completed = 'You genotools-api job has been completed, Please check logs for details.'
sender_email = "si11080772@gmail.com"

pat_KEY = os.environ.get("PAT_TOKEN")
pat_KEY = "qgetcdycqfkqpiaq"

# import asyncio


load_dotenv()


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def send_email(subject, body, sender_email, pat_KEY, recipient_email):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, pat_KEY)
    text = msg.as_string()
    server.sendmail(sender_email, recipient_email, text)
    server.quit()


# def access_secret_version():
#     client = secretmanager.SecretManagerServiceClient()
#     secret_name = f"projects/776926281950/secrets/genotools-api-key/versions/latest"
#     response = client.access_secret_version(name=secret_name)
#     return response.payload.data.decode("UTF-8")

# API_KEY = access_secret_version()

#Using this method for now, will get from secret manager later.
API_KEY = os.environ.get("API_TOKEN")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    return api_key

router = APIRouter()


def background_task(command, recipient):
    """
    Simulates a long-running background task.
    """
    try:    
        logger.info(f"Starting background task")
        result = execute_genotools(command, run_locally=True)
        # sleep(60)
        send_email(subject_completed, body_completed+f"\n\nJob ID: {os.getpid()} - Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"+"\n\nSubmission Command: "+command+"\n\nResults: "+result, sender_email, pat_KEY, recipient)
        logger.info(f"Completed background task and Email Sent")
    except Exception as e:
        logger.error(f"Error Submitting background task: {e}")
        send_email(subject_completed, body_completed+f"\n\nJob ID: {os.getpid()} - Failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"+"\n\nSubmission Command: "+command+"\n\Error: "+{e}, sender_email, pat_KEY, recipient)


@router.get("/")
async def root():
    return "Welcome to GenoTools"

# @router.post("/run-genotools/",dependencies=[Depends(get_api_key)])
@router.post("/run-genotools/")
# def run_genotools(params: GenoToolsParams, api_key: APIKey = Depends(get_api_key)):
async def run_genotools(params: GenoToolsParams, background_tasks: BackgroundTasks, api_key: APIKey = Depends(get_api_key)):#, api_key: APIKey = Depends(get_api_key)):    
    logger.debug(f"Received payload: {params}")
    recipient = params.email

    params.pfile = f'/app/genotools_api/data/{params.pfile}'
    params.ref_panel = f'/app/genotools_api/data/{params.ref_panel}'
    params.ref_labels = f'/app/genotools_api/data/{params.ref_labels}'
    params.model = f'/app/genotools_api/data/{params.model}'

    os.makedirs(f"/app/genotools_api/data/{'/'.join(params.out.split('/')[:-1])}/", exist_ok=True)
    params.out = f'/app/genotools_api/data/{params.out}'
    

    command = construct_command(params)

    send_email(subject_submitted, body_submitted+f"\n\nJob ID: {os.getpid()} - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"+"\n\nSubmission Command: "+command, sender_email, pat_KEY, recipient)      
    # Add the background task to the queue
    background_tasks.add_task(background_task, command, recipient)
    return {
        "message": "Job submitted",
        "command": command
        # "result": result
        }