import os
from dotenv import load_dotenv
load_dotenv()

# URL to scrape
URL = "https://www.google.com/about/careers/applications/jobs/results"

# Scraping interval in hours
SCRAPE_INTERVAL = 1

# Output file names
JSON_FILE = "Jobs/jobs.json"
CSV_FILE = "Jobs/jobs.csv"

# Logging configuration
LOG_FILE = "job_scraper.log"
LOG_LEVEL = "INFO"

# Email notification settings
SEND_EMAIL_NOTIFICATIONS = True
EMAIL_SENDER = os.getenv('EMAIL_SENDER')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
EMAIL_RECIPIENTS = [
    "phanee.5g@gmail.com",
]
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587