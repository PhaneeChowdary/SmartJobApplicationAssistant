# import streamlit as st
# from typing import List
# import pandas as pd
# from langchain_community.chat_models import ChatOllama
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# from bs4 import BeautifulSoup
# import requests
# import tempfile
# import os

# class ResumeMatchingSystem:
#     def __init__(self, model_name: str = "llama3.2"):
#         # Initialize embeddings
#         self.embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2"
#         )
        
#         # Initialize vector store
#         self.vector_store = Chroma(
#             embedding_function=self.embeddings,
#             persist_directory="./chroma_db"
#         )
        
#         # Model configurations
#         self.model_configs = {
#             "llama3.2": {
#                 "temperature": 0.1,
#                 "num_ctx": 4096,
#                 "stop": ["\n\n"],
#                 "repeat_penalty": 1.1
#             },
#             "llama3.2-vision": {
#                 "temperature": 0.2,
#                 "num_ctx": 8192,
#                 "stop": ["\n\n"],
#                 "repeat_penalty": 1.1
#             },
#             "qwen2.5-coder": {
#                 "temperature": 0.1,
#                 "num_ctx": 4096,
#                 "stop": ["\n\n"],
#                 "repeat_penalty": 1.2
#             }
#         }
        
#         # Initialize selected model
#         try:
#             config = self.model_configs.get(model_name, self.model_configs["llama3.2"])
#             self.llm = ChatOllama(model=model_name, **config)
#         except Exception as e:
#             st.error(f"Error initializing Ollama with model {model_name}")
#             st.error(f"Error details: {str(e)}")
#             st.info("Make sure Ollama is running and the selected model is installed")
#             raise e
    
#     def scrape_job_description(self, url: str) -> str:
#         try:
#             headers = {
#                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#             }
#             response = requests.get(url, headers=headers)
#             soup = BeautifulSoup(response.text, 'html.parser')
            
#             # Try different common job description selectors
#             possible_selectors = [
#                 'div[class*="job-description"]',
#                 'div[class*="description"]',
#                 'div[class*="details"]',
#                 'div[class*="content"]'
#             ]
            
#             job_description = None
#             for selector in possible_selectors:
#                 elements = soup.select(selector)
#                 if elements:
#                     job_description = elements[0].get_text(strip=True, separator='\n')
#                     break
            
#             if not job_description:
#                 # If no specific selectors work, try to get the main content
#                 main_content = soup.find(['main', 'article'])
#                 if main_content:
#                     job_description = main_content.get_text(strip=True, separator='\n')
#                 else:
#                     job_description = soup.get_text(strip=True, separator='\n')
            
#             return job_description
#         except Exception as e:
#             st.error(f"Error scraping job description: {str(e)}")
#             return None

#     def load_resume(self, uploaded_file) -> str:
#         try:
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
#                 tmp_file.write(uploaded_file.getvalue())
#                 tmp_file_path = tmp_file.name

#             loader = PyPDFLoader(tmp_file_path)
#             documents = loader.load()
#             text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#             resume_chunks = text_splitter.split_documents(documents)
            
#             os.unlink(tmp_file_path)
#             return "\n".join([doc.page_content for doc in resume_chunks])
#         except Exception as e:
#             st.error(f"Error processing resume: {str(e)}")
#             return None

#     def analyze_match(self, job_description: str, resume: str, model_name: str) -> str:
#         prompts = {
#             "llama3.2": """
#             You are an expert AI Resume Analyzer with deep understanding of job requirements and candidate qualifications. Your task is to provide a comprehensive analysis of how well the given resume matches the job description.

#             Job Description:
#             {job_description}

#             Resume:
#             {resume}

#             Please provide a detailed analysis following this EXACT structure:

#             # Match Analysis Report

#             ## Overall Match Score
#             - Provide a percentage score (0-100%)
#             - Explain why you gave this score with specific examples

#             ## Key Matching Skills
#             - List and explain each skill that matches well
#             - Provide specific examples from both the resume and job description
#             - Highlight particularly strong matches

#             ## Missing or Underdeveloped Skills
#             - Identify important skills from the job description that are missing or need strengthening
#             - Explain why these skills are important for the role
#             - Suggest ways to develop these skills

#             ## Resume Improvement Recommendations
#             - Provide specific suggestions to enhance the resume
#             - Point out areas that need better emphasis or clarification
#             - Suggest reorganization if needed

#             ## Additional Recommendations
#             - Suggest relevant certifications
#             - Recommend training or courses
#             - Provide tips for interview preparation

#             Remember to:
#             - Be specific and detailed in your analysis
#             - Use bullet points for clarity
#             - Provide actionable recommendations
#             - Support your analysis with examples from both documents

#             Format your response using proper markdown syntax with clear sections and bullet points.
#             """,
            
#             "llama3.2-vision": """
#             You are an expert AI Resume Analyzer with enhanced visual understanding capabilities.
#             Perform a comprehensive analysis of the resume and job description match.
            
#             Job Description:
#             {job_description}
            
#             Resume:
#             {resume}
            
#             Provide a detailed analysis with:
#             1. Overall match score (0-100%)
#             2. Matching skills analysis
#             3. Missing skills analysis
#             4. Visual presentation suggestions
#             5. Career progression alignment
#             6. Recommendations for improvements
            
#             Format the response in a clear, structured way using markdown.
#             """,
            
#             "qwen2.5-coder": """
#             You are an expert Technical Resume Analyzer specializing in software development roles.
#             Perform a technical analysis of the resume for this coding position.
            
#             Job Description:
#             {job_description}
            
#             Resume:
#             {resume}
            
#             Analyze with focus on:
#             1. Technical skills match percentage
#             2. Programming languages and frameworks alignment
#             3. System design and architecture capabilities
#             4. Code quality indicators from experience
#             5. Technical project experience relevance
#             6. Missing technical skills and learning recommendations
#             7. Suggested technical certifications
            
#             Format the response in a clear, structured way using markdown.
#             """
#         }
        
#         try:
#             prompt = ChatPromptTemplate.from_template(
#                 prompts.get(model_name, prompts["llama3.2"])
#             )
#             # Create a more robust chain with error handling
#             chain = prompt | self.llm | StrOutputParser()
            
#             # Add timeout and retry logic
#             try:
#                 analysis = chain.invoke({
#                     "job_description": job_description, 
#                     "resume": resume
#                 }, config={
#                     "timeout": 120,  # 2 minute timeout
#                     "retry_on_failure": True
#                 })
                
#                 # Validate if we got a proper analysis
#                 if len(analysis.strip()) < 100 or "Match Analysis Report" not in analysis:
#                     raise ValueError("Incomplete analysis received")
                    
#             except Exception as e:
#                 st.error(f"Error during analysis: {str(e)}")
#                 # Retry with a simplified prompt
#                 simplified_prompt = ChatPromptTemplate.from_template("""
#                 Analyze this resume against the job description. Include:
#                 1. Match percentage
#                 2. Matching skills
#                 3. Missing skills
#                 4. Improvements needed
                
#                 Job Description: {job_description}
#                 Resume: {resume}
#                 """)
#                 chain = simplified_prompt | self.llm | StrOutputParser()
#                 analysis = chain.invoke({
#                     "job_description": job_description, 
#                     "resume": resume
#                 })
#             return analysis
#         except Exception as e:
#             st.error(f"Error analyzing match: {str(e)}")
#             return None

# def main():
#     st.set_page_config(page_title="AI Resume Matcher", layout="wide")
    
#     st.title("🎯 AI Resume Matcher")
#     st.write("Upload your resume and provide a job link to get personalized recommendations!")

#     # Model selection
#     model_descriptions = {
#         "llama3.2": "Balanced model good for general resume analysis",
#         "llama3.2-vision": "Advanced model with better understanding of visual elements and layout",
#         "qwen2.5-coder": "Specialized for technical and programming positions"
#     }
    
#     with st.expander("ℹ️ Model Information", expanded=True):
#         st.markdown("""
#         ### Available Models:
#         1. **Llama 3.2** (2.0 GB)
#            - Balanced performance
#            - Good for general resume analysis
#            - Fast response time
        
#         2. **Llama 3.2 Vision** (7.9 GB)
#            - Better at understanding document layout
#            - Enhanced analysis capabilities
#            - Good for design-focused positions
        
#         3. **Qwen 2.5 Coder** (9.0 GB)
#            - Specialized for technical positions
#            - Strong coding knowledge
#            - Best for developer roles
#         """)
    
#     selected_model = st.selectbox(
#         "Select AI Model",
#         options=list(model_descriptions.keys()),
#         format_func=lambda x: f"{x} - {model_descriptions[x]}"
#     )

#     # Initialize the matching system with selected model
#     try:
#         matcher = ResumeMatchingSystem(selected_model)
#     except Exception:
#         st.stop()

#     # Create two columns for input
#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("📄 Upload Your Resume")
#         uploaded_files = st.file_uploader(
#             "Upload one or more resumes (PDF format)",
#             type="pdf",
#             accept_multiple_files=True
#         )

#     with col2:
#         st.subheader("🔗 Job Details")
#         job_url = st.text_input("Enter the job posting URL")
        
#         st.subheader("📝 Or paste job description")
#         job_description_text = st.text_area(
#             "Paste the job description here if you have it",
#             height=200
#         )
        
#     if uploaded_files and (job_url or job_description_text):
#         if st.button(f"Analyze Match using {selected_model}"):
#             with st.spinner(f"Initializing analysis with {selected_model}..."):
#                 progress_bar = st.progress(0)
#                 status_text = st.empty()

#                 # Get job description
#                 job_description = None
#                 if job_url:
#                     status_text.text("Scraping job description...")
#                     progress_bar.progress(20)
#                     job_description = matcher.scrape_job_description(job_url)
#                 if job_description_text:
#                     job_description = job_description_text
                
#                 if job_description:
#                     # Process each resume
#                     for idx, uploaded_file in enumerate(uploaded_files):
#                         status_text.text(f"Processing resume {idx + 1}/{len(uploaded_files)}...")
#                         progress_bar.progress(40 + (idx * 20))
                        
#                         resume_text = matcher.load_resume(uploaded_file)
                        
#                         if resume_text:
#                             status_text.text(f"Analyzing match using {selected_model}...")
#                             progress_bar.progress(80)
#                             analysis = matcher.analyze_match(
#                                 job_description, 
#                                 resume_text,
#                                 selected_model
#                             )
                            
#                             if analysis:
#                                 st.subheader(f"Analysis for {uploaded_file.name}")
#                                 st.markdown(analysis)
                    
#                     progress_bar.progress(100)
#                     status_text.text("Analysis complete!")
                    
#                     progress_bar.empty()
#                     status_text.empty()

# if __name__ == "__main__":
#     main()








import streamlit as st
from typing import List
import pandas as pd
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
            "qwen2.5-coder": {
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
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try different common job description selectors
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
                # If no specific selectors work, try to get the main content
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
            Your response should be comprehensive and well-structured.
            
            Human: Please analyze the following resume and job description:
            
            Job Description:
            {job_description}

            Resume:
            {resume}

            Provide a detailed analysis following this EXACT structure:

            # Match Analysis Report

            ## Overall Match Score
            - Provide a percentage score (0-100%)
            - Explain why you gave this score with specific examples

            ## Key Matching Skills
            - List and explain each skill that matches well
            - Provide specific examples from both the resume and job description
            - Highlight particularly strong matches

            ## Missing or Underdeveloped Skills
            - Identify important skills from the job description that are missing or need strengthening
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
            """,
            
            "llama3.2-vision": """
            You are an expert AI Resume Analyzer with enhanced visual understanding capabilities.
            Perform a comprehensive analysis of the resume and job description match.
            
            Job Description:
            {job_description}
            
            Resume:
            {resume}
            
            Provide a detailed analysis following this EXACT structure:
            
            # Visual Resume Analysis Report

            ## Overall Match Score
            - Calculate and explain a match percentage (0-100%)
            - Justify the score based on specific matches

            ## Document Presentation Analysis
            - Layout and formatting assessment
            - Visual hierarchy evaluation
            - Professional appearance review

            ## Skills and Experience Match
            - Core competencies alignment
            - Experience relevance
            - Project highlights

            ## Areas for Improvement
            - Visual presentation suggestions
            - Content organization recommendations
            - Format and style enhancements

            ## Career Progression
            - Career trajectory analysis
            - Growth potential
            - Role alignment recommendations

            Format your response in a clear, structured way using markdown with proper headings and bullet points.
            """,
            
            "qwen2.5-coder": """
            You are an expert Technical Resume Analyzer specializing in software development roles.
            Perform a technical analysis of the resume for this coding position.
            
            Job Description:
            {job_description}
            
            Resume:
            {resume}
            
            Provide a detailed analysis following this EXACT structure:

            # Technical Resume Analysis Report

            ## Technical Match Score
            - Overall technical skills match percentage
            - Detailed breakdown of match areas

            ## Programming Skills Analysis
            - Languages and frameworks evaluation
            - Technical stack alignment
            - Development methodology experience

            ## System Design & Architecture
            - Architecture experience assessment
            - System design capabilities
            - Scalability and performance considerations

            ## Code Quality & Best Practices
            - Code quality indicators
            - Testing and documentation experience
            - Development process familiarity

            ## Technical Project Experience
            - Key project analysis
            - Technical leadership indicators
            - Problem-solving examples

            ## Growth Areas & Recommendations
            - Missing technical skills
            - Suggested learning paths
            - Recommended certifications

            Format your response in a clear, structured way using markdown with proper headings and bullet points.
            """
        }
        
        try:
            # Get the appropriate prompt for the model
            prompt_template = prompts.get(model_name, prompts["llama3.2"])
            
            # Create the prompt
            prompt = ChatPromptTemplate.from_template(prompt_template)
            
            # Create and run the chain
            chain = prompt | self.llm | StrOutputParser()
            
            try:
                # Get initial analysis
                analysis = chain.invoke({
                    "job_description": job_description, 
                    "resume": resume
                }, config={
                    "timeout": 180,  # 3 minute timeout
                    "retry_on_failure": True
                })
                
                # Validate response and get additional details if needed
                if len(analysis.strip()) < 200:  # Increased minimum length
                    st.warning("Initial analysis was brief, getting more details...")
                    
                    detailed_prompt = ChatPromptTemplate.from_template("""
                    Based on the initial analysis, please provide more specific details about:
                    
                    Previous Analysis:
                    {previous_analysis}
                    
                    Job Description:
                    {job_description}
                    
                    Resume:
                    {resume}
                    
                    Please provide additional details focusing on:
                    1. More specific examples of matching skills
                    2. Detailed explanation of missing skills
                    3. Concrete, actionable resume improvements
                    4. Specific certifications and courses to consider
                    5. Practical steps for skill development
                    
                    Format the response in a clear, structured way using markdown.
                    """)
                    
                    chain = detailed_prompt | self.llm | StrOutputParser()
                    additional_analysis = chain.invoke({
                        "previous_analysis": analysis,
                        "job_description": job_description,
                        "resume": resume
                    })
                    
                    # Combine analyses
                    analysis = f"{analysis}\n\n## Additional Details\n\n{additional_analysis}"
                
                return analysis
                
            except Exception as e:
                st.warning(f"Initial analysis attempt failed ({str(e)}), trying simplified approach...")
                
                # Fallback to simpler prompt
                simplified_prompt = ChatPromptTemplate.from_template("""
                Analyze this resume for the job position:

                Job Description:
                {job_description}

                Resume:
                {resume}

                Provide a clear analysis covering:
                1. Match Percentage (0-100%) with explanation
                2. Key matching skills found
                3. Important missing skills
                4. Specific improvement recommendations

                Keep the response informative but concise.
                Format in markdown with clear sections.
                """)
                
                chain = simplified_prompt | self.llm | StrOutputParser()
                analysis = chain.invoke({
                    "job_description": job_description, 
                    "resume": resume
                })
                
                return analysis
                
        except Exception as e:
            st.error(f"Error in resume analysis: {str(e)}")
            return "Error: Unable to complete the analysis. Please try again or select a different model."

def main():
    st.set_page_config(page_title="AI Resume Matcher", layout="wide")
    
    st.title("🎯 AI Resume Matcher")
    st.write("Upload your resume and provide a job link to get personalized recommendations!")

    # Model selection
    model_descriptions = {
        "llama3.2": "Balanced model good for general resume analysis",
        "llama3.2-vision": "Advanced model with better understanding of visual elements and layout",
        "qwen2.5-coder": "Specialized for technical and programming positions"
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
        
        3. **Qwen 2.5 Coder** (9.0 GB)
           - Specialized for technical positions
           - Strong coding knowledge
           - Best for developer roles
        """)
    
    selected_model = st.selectbox(
        "Select AI Model",
        options=list(model_descriptions.keys()),
        format_func=lambda x: f"{x} - {model_descriptions[x]}"
    )

    # Initialize the matching system with selected model
    try:
        matcher = ResumeMatchingSystem(selected_model)
    except Exception:
        st.stop()

    # Create two columns for input
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

                # Get job description
                job_description = None
                if job_url:
                    status_text.text("Scraping job description...")
                    progress_bar.progress(20)
                    job_description = matcher.scrape_job_description(job_url)
                if job_description_text:
                    job_description = job_description_text
                
                if job_description:
                    # Process each resume
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
                                st.markdown(analysis)
                    
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    
                    progress_bar.empty()
                    status_text.empty()


if __name__ == "__main__":
    main()