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

## 📥 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/PhaneeChowdary/SmartJobApplicationAssistant
   cd SmartJobApplicationAssistant
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

## 🚀 Usage

1. Configure your settings in `config.py`

2. Run the job scraper:
   ```bash
   python scraper.py
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

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎯 Future Enhancements

- AI-powered resume optimization
- Multiple job portal support
- Interview preparation suggestions
- Application tracking system
- Success rate analytics

---
## Thank you