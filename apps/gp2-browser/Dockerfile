#Base Image to use
FROM python:3.10.8

RUN apt update
RUN apt install git

#Expose port 8080
EXPOSE 8080

#Copy Requirements.txt file into app directory
COPY requirements.txt app/requirements.txt

#install all requirements in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r app/requirements.txt

# RUN git clone https://github.com/dvitale199/GenoTools && cd GenoTools && pip install .

#Copy all files in current directory into app directory
COPY . /app

#Change Working Directory to app directory
WORKDIR /app

#Run the application on port 8080
ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8080", "--server.address=0.0.0.0"]
