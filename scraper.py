from bs4 import BeautifulSoup
import csv, json, schedule, time, logging, sys, smtplib, requests
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import (URL, SCRAPE_INTERVAL, JSON_FILE, CSV_FILE, LOG_FILE, LOG_LEVEL,
                    SEND_EMAIL_NOTIFICATIONS, EMAIL_SENDER, EMAIL_PASSWORD,
                    EMAIL_RECIPIENTS, SMTP_SERVER, SMTP_PORT)
import time
import requests
import logging

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

MAX_RETRIES = 5  # Number of retries
BACKOFF_FACTOR = 2  # Exponential backoff factor
TIMEOUT = 10  # Timeout for each request in seconds

def fetch_webpage(url):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = requests.get(url, timeout=TIMEOUT)
            response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
            return response.text
        except requests.RequestException as e:
            retries += 1
            wait_time = BACKOFF_FACTOR ** retries  # Exponential backoff
            logging.error(f"Error fetching webpage (attempt {retries}): {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    logging.error(f"Failed to fetch webpage after {MAX_RETRIES} attempts. Skipping this update.")
    return None


def send_email_notification(new_jobs):
    if not SEND_EMAIL_NOTIFICATIONS:
        return

    subject = f"New Job Alerts - {len(new_jobs)} new jobs found"
    
    def job_listing_html(job):
        qualifications_html = ''.join(f'<li>{qual}</li>' for qual in job['Minimum Qualifications'])
        return f"""
        <hr>
        <div class="card">
            <h2><a href="{job['Apply Link']}">{job['Title']}</a></h2>
            <p><strong>Location:</strong> {job['Location']}</p>
            <p><strong>Experience Level:</strong> {job['Experience Level']}</p>
            <div class="qualifications">
                <strong>Qualifications:</strong>
                <ul>
                    {qualifications_html}
                </ul>
            </div>
            <a href="{job['Apply Link']}" class="apply-button">View/Apply</a>
        </div>
        """

    job_listings_html = ''.join(job_listing_html(job) for job in new_jobs)

    html_content = f"""
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: 'Times New Roman';
                line-height: 1.6;
                color: #202124;
                background-color: #f1f3f4;
                margin: 10px;
                padding: 20px;
            }}
            .container {{
                max-width: 600px;
                margin: 0 auto;
                background-color: #ffffff;
                border-radius: 8px;
                overflow: hidden;
            }}
            .header {{
                background-color: #1a73e8;
                color: #ffffff;
                padding: 20px;
                text-align: center;
            }}
            h1 {{
                margin: 0;
                font-size: 24px;
                font-weight: normal;
            }}
            .content {{
                padding: 20px;
            }}
            .card {{
                background-color: #ffffff;
                border-radius: 25px;
                padding: 12px;
                margin:10px;
                margin-bottom: 16px;
            }}
            .card h2 {{
                color: #1a73e8;
                font-size: 18px;
                margin-top: 0;
                margin-bottom: 12px;
            }}
            .card p {{
                margin: 8px 0;
            }}
            .qualifications {{
                margin-top: 12px;
            }}
            .qualifications ul {{
                margin: 8px 0;
                padding-left: 20px;
            }}
            .apply-button {{
                display: inline-block;
                background-color: #1a73e8;
                color: #ffffff;
                padding: 8px 16px;
                text-decoration: none;
                border-radius: 25px;
            }}
            .footer {{
                text-align: center;
                padding: 20px;
                color: #5f6368;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>New Job Alerts</h1>
            </div>
            <div class="content">
                <p>We've found {len(new_jobs)} new job openings that might interest you:</p>
                
                {job_listings_html}
            </div>
            <div class="footer">
                <p>Good luck with your job search!</p>
            </div>
        </div>
    </body>
    </html>
    """

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            
            for recipient in EMAIL_RECIPIENTS:
                msg = MIMEMultipart('alternative')
                msg['From'] = EMAIL_SENDER
                msg['To'] = recipient
                msg['Subject'] = subject

                text_content = "New jobs found. Please view this email in an HTML-capable email client."
                msg.attach(MIMEText(text_content, 'plain'))
                msg.attach(MIMEText(html_content, 'html'))
                
                server.send_message(msg)
                logging.info(f"Email notification sent successfully to {recipient}")
                
        logging.info("All email notifications sent successfully")
    except Exception as e:
        logging.error(f"Failed to send email notifications: {e}")


def scrape_jobs(html_content):
    if not html_content:
        return []
    
    soup = BeautifulSoup(html_content, 'html.parser')
    job_listings = soup.find_all('li', class_='lLd3Je')
    all_jobs = []

    for job in job_listings:
        try:
            title = job.find('h3', class_='QJPWVe').text.strip()
            location = job.find('span', class_='r0wTof').text.strip()
            
            experience_span = job.find('span', class_='wVSTAb')
            experience = experience_span.text.strip() if experience_span else "Not specified"
            
            qual_div = job.find('div', class_='Xsxa1e')
            qualifications = [li.text.strip() for li in qual_div.find_all('li')] if qual_div else []
            
            link_elem = job.find('a', class_='WpHeLc')
            if link_elem and 'href' in link_elem.attrs:
                link = "https://www.google.com/about/careers/applications/jobs/results/" + link_elem['href'].split('/')[-1]
            else:
                link = "No link available"

            job_data = {
                "Title": title,
                "Location": location,
                "Experience Level": experience,
                "Minimum Qualifications": qualifications,
                "Apply Link": link,
                "First Seen": datetime.now().isoformat()
            }
            
            all_jobs.append(job_data)
        except AttributeError as e:
            logging.warning(f"Error parsing job listing: {e}")
            continue

    return all_jobs


def load_existing_jobs():
    try:
        with open(JSON_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.info(f"No existing job file found. Starting fresh.")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding existing job file: {e}")
        return []

def update_jobs(new_jobs, existing_jobs):
    updated_jobs = existing_jobs.copy()
    new_job_listings = []
    for new_job in new_jobs:
        if not any(existing_job['Apply Link'] == new_job['Apply Link'] for existing_job in existing_jobs):
            updated_jobs.append(new_job)
            new_job_listings.append(new_job)
    return updated_jobs, new_job_listings

def export_to_json(jobs, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as json_file:
            json.dump(jobs, json_file, ensure_ascii=False, indent=4)
        logging.info(f"JSON file '{filename}' has been updated.")
    except IOError as e:
        logging.error(f"Error writing to JSON file: {e}")


def export_to_csv(jobs, filename):
    csv_fields = ["Title", "Location", "Experience Level", "Minimum Qualifications", "Apply Link", "First Seen"]
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
            writer.writeheader()
            for job in jobs:
                job_csv = job.copy()
                job_csv["Minimum Qualifications"] = "; ".join(job["Minimum Qualifications"])
                writer.writerow(job_csv)
        logging.info(f"CSV file '{filename}' has been updated.")
    except IOError as e:
        logging.error(f"Error writing to CSV file: {e}")


def scrape_and_update():
    logging.info(f"Running job scraper at {datetime.now().isoformat()}")
    html_content = fetch_webpage(URL)
    if html_content:
        new_jobs = scrape_jobs(html_content)
        existing_jobs = load_existing_jobs()
        updated_jobs, new_job_listings = update_jobs(new_jobs, existing_jobs)
        export_to_json(updated_jobs, JSON_FILE)
        export_to_csv(updated_jobs, CSV_FILE)
        logging.info(f"Found {len(new_job_listings)} new job postings")
        if new_job_listings:
            send_email_notification(new_job_listings)
    else:
        logging.error("Failed to fetch webpage. Skipping this update.")

def main():
    logging.info("Starting job scraper")
    
    # Run immediately on start
    scrape_and_update()
    
    logging.info(f"Setting up schedule to run every {SCRAPE_INTERVAL} hours")
    # Schedule to run at the specified interval
    schedule.every(SCRAPE_INTERVAL).hours.do(scrape_and_update)
    
    logging.info("Entering main loop")
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Job scraper stopped by user")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()