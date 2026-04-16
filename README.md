# Make Up Recommendation and E-Commerce Website

## Description
## Key Features
## Tech Stack
## Screenshots
## Setup Instruction
1. Clone the project
   * open terminal in IDE (VS Code)
   * command:
2. create virtual environment command
   * python -m venv venv
3. Activate venv command
   * windows: venv\Scripts\activate
   * mac: source venv/bin/activate
4. Install dependencies command
   * pip install -r requirements.txt
5. Open MySQL Workbench or terminal and create database
   * command: CREATE DATABASE ecomakeup;
6. Update database information in file settings.py
7. Install MySQL Driver
   * command: pip install mysqlclient
8. Build database structure
   * command: python manage.py migrate
9. Create superuser to access admin dashboard
   * command: python manage.py createsuperuser
   * enter username and password created at login page to redirect to admin dashboard
9. Run the server
   * command: python manage.py runserver

**Developed by Ezlyn Azwa**
