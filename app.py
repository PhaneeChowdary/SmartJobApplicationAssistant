import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from bs4 import BeautifulSoup
import requests
import tempfile
import os
import plotly.graph_objects as go
import re

def create_radar_chart(skills_data):
    """Create a radar chart for skills matching."""
    categories = [skill['name'] for skill in skills_data]
    scores = [skill['score'] for skill in skills_data]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Skills Match'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="Skills Match Analysis"
    )
    
    return fig

def create_category_bar_chart(overall_score, skills_score, experience_score):
    """Create a bar chart for different score categories."""
    categories = ['Overall Match', 'Skills Match', 'Experience Match']
    scores = [overall_score, skills_score, experience_score]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=scores,
            marker_color=['#3366cc', '#dc3912', '#ff9900']
        )
    ])
    
    fig.update_layout(
        title="Match Score Breakdown",
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 100]),
        showlegend=False
    )
    
    return fig

def create_gauge_chart(score):
    """Create a gauge chart for overall match score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Match Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2ecc71" if score >= 75 else "#f1c40f" if score >= 50 else "#e74c3c"},
            'steps': [
                {'range': [0, 50], 'color': "rgba(231, 76, 60, 0.2)"},
                {'range': [50, 75], 'color': "rgba(241, 196, 15, 0.2)"},
                {'range': [75, 100], 'color': "rgba(46, 204, 113, 0.2)"}
            ]
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def extract_percentage(text):
    """Extract percentage from text."""
    match = re.search(r'(\d+)%', text)
    if match:
        return int(match.group(1))
    match = re.search(r'(\d+)', text)
    if match:
        return int(match.group(1))
    return 0

def parse_analysis_scores(analysis_text):
    """Parse the analysis text to extract scores."""
    lines = analysis_text.split('\n')
    overall_score = 0
    skills_score = 0
    experience_score = 0
    skills_data = []
    
    section = ""
    for line in lines:
        if '# ' in line or '## ' in line:
            section = line.lower()
            continue
            
        if 'overall match score' in section.lower() and '%' in line:
            overall_score = extract_percentage(line)
            
        if 'matching skills' in section.lower() and '-' in line:
            skill_match = extract_percentage(line)
            skill_name = line.split('-')[1].split(':')[0].strip() if ':' in line else line.split('-')[1].strip()
            if skill_match > 0:
                skills_data.append({"name": skill_name, "score": skill_match})
                
        if 'experience' in line.lower() and '%' in line:
            experience_score = extract_percentage(line)
    
    if not skills_data:
        # Create default skills data if none found
        skills_data = [
            {"name": "Technical Skills", "score": overall_score},
            {"name": "Experience", "score": experience_score if experience_score > 0 else overall_score - 10},
            {"name": "Education", "score": overall_score - 5},
            {"name": "Communication", "score": overall_score + 5 if overall_score <= 95 else 100}
        ]
    
    return overall_score, skills_data, experience_score if experience_score > 0 else overall_score - 15

def display_visualizations(analysis_text):
    """Display all visualizations for the analysis."""
    overall_score, skills_data, experience_score = parse_analysis_scores(analysis_text)
    skills_avg = sum(skill['score'] for skill in skills_data) / len(skills_data)
    
    # Create three columns for the visualizations
    col1, col2 = st.columns(2)
    
    # Display gauge chart in its own row
    st.plotly_chart(create_gauge_chart(overall_score), use_container_width=True)
    
    with col1:
        # Display radar chart
        st.plotly_chart(create_radar_chart(skills_data), use_container_width=True)
    
    with col2:
        # Display category bar chart
        st.plotly_chart(create_category_bar_chart(overall_score, skills_avg, experience_score), use_container_width=True)

class ResumeMatchingSystem:
    def __init__(self, model_name: str = "llama3.2"):
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize vector store
        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        # Model configurations
        self.model_configs = {
            "llama3.2": {
                "temperature": 0.7,
                "num_ctx": 8192,
                "num_predict": 2048,
                "repeat_penalty": 1.1,
                "top_k": 50,
                "top_p": 0.9
            },
            "llama3.2-vision": {
                "temperature": 0.7,
                "num_ctx": 8192,
                "num_predict": 2048,
                "repeat_penalty": 1.1,
                "top_k": 50,
                "top_p": 0.9
            },
            "qwen2.5-coder:14b": {
                "temperature": 0.7,
                "num_ctx": 8192,
                "num_predict": 2048,
                "repeat_penalty": 1.1,
                "top_k": 50,
                "top_p": 0.9
            }
        }
        
        # Initialize selected model
        try:
            config = self.model_configs.get(model_name, self.model_configs["llama3.2"])
            self.llm = ChatOllama(model=model_name, **config)
        except Exception as e:
            st.error(f"Error initializing Ollama with model {model_name}")
            st.error(f"Error details: {str(e)}")
            st.info("Make sure Ollama is running and the selected model is installed")
            raise e
    
    def scrape_job_description(self, url: str) -> str:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            possible_selectors = [
                'div[class*="job-description"]',
                'div[class*="description"]',
                'div[class*="details"]',
                'div[class*="content"]'
            ]
            
            job_description = None
            for selector in possible_selectors:
                elements = soup.select(selector)
                if elements:
                    job_description = elements[0].get_text(strip=True, separator='\n')
                    break
            
            if not job_description:
                main_content = soup.find(['main', 'article'])
                if main_content:
                    job_description = main_content.get_text(strip=True, separator='\n')
                else:
                    job_description = soup.get_text(strip=True, separator='\n')
            
            return job_description
        except Exception as e:
            st.error(f"Error scraping job description: {str(e)}")
            return None

    def load_resume(self, uploaded_file) -> str:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            resume_chunks = text_splitter.split_documents(documents)
            
            os.unlink(tmp_file_path)
            return "\n".join([doc.page_content for doc in resume_chunks])
        except Exception as e:
            st.error(f"Error processing resume: {str(e)}")
            return None

    def analyze_match(self, job_description: str, resume: str, model_name: str) -> str:
        prompts = {
            "llama3.2": """
            System: You are an expert AI Resume Analyzer. Provide a detailed analysis of the resume match.
            Your response should be well-structured and include specific percentages for different aspects.
            
            Human: Please analyze the following resume and job description:
            
            Job Description:
            {job_description}

            Resume:
            {resume}

            Provide a detailed analysis following this EXACT structure:

            # Match Analysis Report

            ## Overall Match Score
            - Overall match: XX% (provide specific percentage)
            - Experience match: XX% (provide specific percentage)
            - Skills match: XX% (provide specific percentage)
            - Explain why you gave these scores with specific examples

            ## Key Matching Skills
            - List each matching skill with a percentage match
            - Provide specific examples from both the resume and job description
            - Highlight particularly strong matches

            ## Missing or Underdeveloped Skills
            - Identify important skills from the job description that are missing
            - Explain why these skills are important for the role
            - Suggest ways to develop these skills

            ## Resume Improvement Recommendations
            - Provide specific suggestions to enhance the resume
            - Point out areas that need better emphasis or clarification
            - Suggest reorganization if needed

            ## Additional Recommendations
            - Suggest relevant certifications
            - Recommend training or courses
            - Provide tips for interview preparation

            Format your response using proper markdown with clear sections and bullet points.
            BE SURE TO INCLUDE PERCENTAGE SCORES FOR OVERALL MATCH AND INDIVIDUAL SKILLS.
            """,
            
            "llama3.2-vision": """
            You are an expert AI Resume Analyzer. Follow the same format as llama3.2 prompt but with additional focus on visual aspects.
            """,
            
            "qwen2.5-coder:14b": """
            You are an expert Technical Resume Analyzer. You already have the resume and job description - analyze them directly.
            
            Job Description:
            {job_description}
            
            Resume:
            {resume}
            
            Provide a detailed technical analysis following this EXACT structure:

            # Technical Resume Analysis Report

            ## Technical Match Score
            - Provide an overall match percentage (0-100%)
            - Break down the scoring by technical areas
            - Explain the scoring rationale

            ## Programming Skills Analysis
            - List all programming languages found with proficiency levels
            - Framework and library expertise
            - Development tools and environments
            - Each skill should include a percentage match score

            ## Technical Project Experience
            - Analyze relevant technical projects
            - Evaluate complexity and scale
            - Assess problem-solving approaches
            - Include percentage match for project experience

            ## System Design & Architecture
            - Evaluate system design experience
            - Assess scalability considerations
            - Review architectural patterns used
            - Include percentage match for design skills

            ## Areas for Technical Growth
            - Identify missing technical skills
            - Suggest specific learning paths
            - Recommend certifications
            - Prioritize improvements needed

            Format your response using proper markdown with clear sections and bullet points.
            ENSURE TO INCLUDE PERCENTAGE SCORES FOR ALL EVALUATIONS.
            """
        }
        
        try:
            prompt_template = prompts.get(model_name, prompts["llama3.2"])
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | self.llm | StrOutputParser()
            
            try:
                analysis = chain.invoke({
                    "job_description": job_description, 
                    "resume": resume
                }, config={
                    "timeout": 180,
                    "retry_on_failure": True
                })
                
                if len(analysis.strip()) < 200:
                    st.warning("Initial analysis was brief, getting more details...")
                    detailed_prompt = ChatPromptTemplate.from_template("""
                    Based on the initial analysis, please provide more specific details and numerical scores...
                    Previous Analysis: {previous_analysis}
                    Job Description: {job_description}
                    Resume: {resume}
                    """)
                    
                    chain = detailed_prompt | self.llm | StrOutputParser()
                    additional_analysis = chain.invoke({
                        "previous_analysis": analysis,
                        "job_description": job_description,
                        "resume": resume
                    })
                    
                    analysis = f"{analysis}\n\n## Additional Details\n\n{additional_analysis}"
                
                return analysis
                
            except Exception as e:
                st.warning(f"Initial analysis attempt failed ({str(e)}), trying simplified approach...")
                # Fallback to simpler prompt...
                simplified_prompt = ChatPromptTemplate.from_template("""
                Provide a simplified analysis with clear percentage scores for overall match and skills...
                """)
                chain = simplified_prompt | self.llm | StrOutputParser()
                return chain.invoke({"job_description": job_description, "resume": resume})
                
        except Exception as e:
            st.error(f"Error in resume analysis: {str(e)}")
            return "Error: Unable to complete the analysis. Please try again or select a different model."

def main():
    st.set_page_config(page_title="AI Resume Matcher", layout="wide")
    
    st.title("🎯 AI Resume Matcher")
    st.write("Upload your resume and provide a job link to get personalized recommendations!")

    model_descriptions = {
        "llama3.2": "Balanced model good for general resume analysis",
        "llama3.2-vision": "Advanced model with better understanding of visual elements and layout",
        "qwen2.5-coder:14b": "Specialized for technical and programming positions"
    }
    
    with st.expander("ℹ️ Model Information", expanded=True):
        st.markdown("""
        ### Available Models:
        1. **Llama 3.2** (2.0 GB)
           - Balanced performance
           - Good for general resume analysis
           - Fast response time
        
        2. **Llama 3.2 Vision** (7.9 GB)
           - Better at understanding document layout
           - Enhanced analysis capabilities
           - Good for design-focused positions
        
        3. **qwen2.5-coder:14b** (9.0 GB)
           - Specialized for technical positions
           - Strong coding knowledge
           - Best for developer roles
        """)
    
    selected_model = st.selectbox(
        "Select AI Model",
        options=list(model_descriptions.keys()),
        format_func=lambda x: f"{x} - {model_descriptions[x]}"
    )

    try:
        matcher = ResumeMatchingSystem(selected_model)
    except Exception:
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Upload Your Resume")
        uploaded_files = st.file_uploader(
            "Upload one or more resumes (PDF format)",
            type="pdf",
            accept_multiple_files=True
        )

    with col2:
        st.subheader("🔗 Job Details")
        job_url = st.text_input("Enter the job posting URL")
        
        st.subheader("📝 Or paste job description")
        job_description_text = st.text_area(
            "Paste the job description here if you have it",
            height=200
        )
        
    if uploaded_files and (job_url or job_description_text):
        if st.button(f"Analyze Match using {selected_model}"):
            with st.spinner(f"Initializing analysis with {selected_model}..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                job_description = None
                if job_url:
                    status_text.text("Scraping job description...")
                    progress_bar.progress(20)
                    job_description = matcher.scrape_job_description(job_url)
                if job_description_text:
                    job_description = job_description_text
                
                if job_description:
                    for idx, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing resume {idx + 1}/{len(uploaded_files)}...")
                        progress_bar.progress(40 + (idx * 20))
                        
                        resume_text = matcher.load_resume(uploaded_file)
                        
                        if resume_text:
                            status_text.text(f"Analyzing match using {selected_model}...")
                            progress_bar.progress(80)
                            analysis = matcher.analyze_match(
                                job_description, 
                                resume_text,
                                selected_model
                            )
                            
                            if analysis:
                                st.subheader(f"Analysis for {uploaded_file.name}")
                                # Display visualizations first
                                display_visualizations(analysis)
                                # Then display the text analysis
                                st.markdown("## Detailed Analysis")
                                st.markdown(analysis)
                    
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    
                    progress_bar.empty()
                    status_text.empty()

if __name__ == "__main__":
    main()