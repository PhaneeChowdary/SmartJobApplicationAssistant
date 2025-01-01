# Smart Job Application Assistant

An intelligent job application system that scrapes job postings and helps create tailored resumes based on your project portfolio and job requirements.

## 🌟 Features

- Automated job scraping from specified job portals
- AI-powered resume generation using Large Language Models
- Smart matching algorithm with percentage-based compatibility scoring
- Intelligent resume tailoring based on job descriptions
- Project-job matching algorithm
- Email notifications for new job matches
- Export data in both JSON and CSV formats
- Configurable scraping intervals
- Robust error handling and retry mechanisms

## 🛠️ Technologies Used

- Python 3.8+
- Large Language Models for resume generation and optimization
- BeautifulSoup4 for web scraping
- Schedule for task automation
- SMTP for email notifications
- Logging for monitoring and debugging
- NLP libraries for text similarity matching

## 📋 Prerequisites

Before running this project, make sure you have:

- Python 3.8 or higher installed
- Access to an SMTP server for email notifications
- Your project portfolio prepared in the required format

## ⚙️ Configuration

Create a `config.py` file with the following parameters:

```python
# Job Portal Configuration
URL = "your_job_portal_url"
SCRAPE_INTERVAL = 6  # Hours

# File Paths
JSON_FILE = "jobs.json"
CSV_FILE = "jobs.csv"
LOG_FILE = "scraper.log"

# Logging Configuration
LOG_LEVEL = "INFO"

# Email Configuration
SEND_EMAIL_NOTIFICATIONS = True
EMAIL_SENDER = "your_email@example.com"
EMAIL_PASSWORD = "your_email_password"
EMAIL_RECIPIENTS = ["recipient@example.com"]
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Project Portfolio Configuration
PROJECTS_FILE = "projects.json"  # Add your projects here

# LLM Configuration
LLM_API_KEY = "your_api_key_here"
LLM_MODEL = "gpt-4"  # or your preferred model
MATCH_THRESHOLD = 75  # Minimum matching percentage for notifications
```

## 📥 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/smart-job-assistant.git
   cd smart-job-assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your project portfolio:
   Create a `projects.json` file with your projects in the following format:
   ```json
   {
     "projects": [
       {
         "title": "Project Name",
         "description": "Detailed description",
         "technologies": ["Python", "Machine Learning", "AWS"],
         "key_achievements": ["Achievement 1", "Achievement 2"],
         "github_link": "https://github.com/..."
       }
     ]
   }
   ```

## 🚀 Usage

1. Configure your settings in `config.py`

2. Run the job scraper:
   ```bash
   python main.py
   ```

The system will:
- Start scraping jobs based on the configured interval
- Match jobs with your project portfolio
- Send email notifications for suitable matches
- Store all job data in both JSON and CSV formats

## 📧 Email Notifications

The system sends beautifully formatted HTML emails containing:
- Job title and location
- Experience level requirements
- Minimum qualifications
- Direct application link
- Matching percentage score
- AI-generated tailored resume
- Matched projects from your portfolio
- Key skills alignment analysis
- Suggested resume modifications

## 📊 Data Storage

Jobs are stored in two formats:
- `jobs.json`: Complete job data in JSON format
- `jobs.csv`: Tabular format for easy viewing and analysis

## 🔍 Logging

The system maintains detailed logs in `scraper.log`, including:
- Scraping activities
- Job matches
- Email notifications
- Errors and retry attempts

## 🛡️ Error Handling

The system includes:
- Exponential backoff for failed requests
- Multiple retry attempts
- Comprehensive error logging
- Graceful failure handling

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎯 Future Enhancements

- AI-powered resume optimization
- Multiple job portal support
- Interview preparation suggestions
- Application tracking system
- Success rate analytics

## ⚠️ Disclaimer

This tool is meant to assist in job applications and should be used responsibly and in accordance with the target job portal's terms of service.