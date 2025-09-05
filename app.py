import streamlit as st
import os
from dotenv import load_dotenv
from src.pdf_extractor import PDFExtractor
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import time

# Load environment variables
load_dotenv()

# Initialize the sentence transformer model globally
@st.cache_resource
def load_sentence_transformer():
    """Load and cache the sentence transformer model"""
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def calculate_semantic_similarity(resume_text, job_description):
    """Calculate semantic similarity using sentence transformers"""
    try:
        # Load the model
        model = load_sentence_transformer()
        
        # Split texts into sentences for better analysis
        resume_sentences = [sent.strip() for sent in resume_text.split('.') if len(sent.strip()) > 10]
        jd_sentences = [sent.strip() for sent in job_description.split('.') if len(sent.strip()) > 10]
        
        # If texts are too short, use the full text
        if len(resume_sentences) < 3:
            resume_sentences = [resume_text]
        if len(jd_sentences) < 3:
            jd_sentences = [job_description]
        
        # Generate embeddings
        resume_embeddings = model.encode(resume_sentences)
        jd_embeddings = model.encode(jd_sentences)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(resume_embeddings, jd_embeddings)
        
        # Get the maximum similarity for each resume sentence
        max_similarities = np.max(similarity_matrix, axis=1)
        
        # Calculate overall similarity score
        overall_similarity = np.mean(max_similarities)
        
        # Convert to percentage (0-100)
        percentage_score = int(overall_similarity * 100)
        
        return max(10, min(95, percentage_score))  # Cap between 10-95%
        
    except Exception as e:
        st.error(f"Error in semantic analysis: {str(e)}")
        return calculate_keyword_fallback_score(resume_text, job_description)

def calculate_keyword_fallback_score(resume_text, job_description):
    """Fallback keyword-based scoring if semantic analysis fails"""
    try:
        # Convert to lowercase for comparison
        resume_lower = resume_text.lower()
        jd_lower = job_description.lower()
        
        # Define comprehensive skill categories
        technical_skills = [
            'python', 'java', 'javascript', 'react', 'node', 'sql', 'mongodb', 'aws', 'docker', 
            'kubernetes', 'git', 'html', 'css', 'angular', 'vue', 'php', 'c++', 'c#', '.net',
            'spring', 'django', 'flask', 'express', 'mysql', 'postgresql', 'redis', 'elasticsearch',
            'machine learning', 'ai', 'data science', 'tensorflow', 'pytorch', 'pandas', 'numpy',
            'api', 'rest', 'graphql', 'microservices', 'devops', 'ci/cd', 'jenkins', 'terraform',
            'azure', 'gcp', 'linux', 'windows', 'android', 'ios', 'swift', 'kotlin', 'flutter',
            'blockchain', 'solidity', 'web3', 'rust', 'go', 'scala', 'r', 'matlab', 'tableau'
        ]
        
        soft_skills = [
            'leadership', 'communication', 'teamwork', 'problem solving', 'analytical', 
            'project management', 'agile', 'scrum', 'collaboration', 'mentoring', 'creativity',
            'adaptability', 'time management', 'critical thinking', 'decision making'
        ]
        
        # Extract keywords from job description
        jd_technical = [skill for skill in technical_skills if skill in jd_lower]
        jd_soft = [skill for skill in soft_skills if skill in jd_lower]
        
        # Check matches in resume
        resume_technical = [skill for skill in jd_technical if skill in resume_lower]
        resume_soft = [skill for skill in jd_soft if skill in resume_lower]
        
        # Calculate scores
        technical_score = (len(resume_technical) / max(len(jd_technical), 1)) * 100 if jd_technical else 60
        soft_score = (len(resume_soft) / max(len(jd_soft), 1)) * 100 if jd_soft else 70
        
        # Weight the scores
        weighted_score = (technical_score * 0.7 + soft_score * 0.3)
        
        return max(15, min(90, int(weighted_score)))
        
    except Exception:
        return 50  # Default fallback score

def extract_key_sections(text):
    """Extract key sections from resume text"""
    sections = {
        'skills': [],
        'experience': [],
        'education': []
    }
    
    # Simple section extraction based on common keywords
    lines = text.split('\n')
    current_section = None
    
    for line in lines:
        line_lower = line.lower().strip()
        
        if any(keyword in line_lower for keyword in ['skill', 'technical', 'programming']):
            current_section = 'skills'
        elif any(keyword in line_lower for keyword in ['experience', 'work', 'employment', 'career']):
            current_section = 'experience'
        elif any(keyword in line_lower for keyword in ['education', 'degree', 'university', 'college']):
            current_section = 'education'
        
        if current_section and len(line.strip()) > 5:
            sections[current_section].append(line.strip())
    
    return sections

def get_score_feedback(score):
    """Get feedback message based on score"""
    if score >= 80:
        return "🎉 Excellent match! Your resume aligns very well with this job.", "success", "#28a745", "#d4edda"
    elif score >= 65:
        return "👍 Good match! Consider highlighting relevant skills more prominently.", "info", "#17a2b8", "#d1ecf1"
    elif score >= 45:
        return "⚠️ Moderate match. You may want to tailor your resume for this role.", "warning", "#ffc107", "#fff3cd"
    elif score >= 25:
        return "❌ Low match. Consider significant resume adjustments for this position.", "error", "#dc3545", "#f8d7da"
    else:
        return "🔴 Very low match. This role may not be suitable or resume needs major updates.", "error", "#dc3545", "#f8d7da"

def add_custom_css():
    """Add custom CSS for better styling"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    /* Card Styling */
    .upload-card, .jd-card, .analyze-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid #e1e8ed;
        transition: all 0.3s ease;
    }
    
    .upload-card:hover, .jd-card:hover, .analyze-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .step-number {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Progress Bar Styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* File Uploader Styling */
    .uploadedFile {
        background: #f8f9fa;
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    
    /* Text Area Styling */
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #e1e8ed;
        font-family: 'Inter', sans-serif;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Results Styling */
    .results-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin-top: 2rem;
    }
    
    /* Footer Styling */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #6c757d;
        font-size: 0.9rem;
        border-top: 1px solid #e1e8ed;
        margin-top: 3rem;
    }
    
    /* Animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Success/Error Messages */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Hide Streamlit Menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="AI Resume Analyzer",
        page_icon="📄",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Add custom CSS
    add_custom_css()
    
    # Header
    st.markdown("""
    <div class="main-header fade-in">
        <div class="main-title">📄 AI Resume Analyzer</div>
        <div class="main-subtitle">Advanced semantic matching powered by sentence-transformers</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content container
    with st.container():
        # Step 1: Resume Upload
        st.markdown("""
        <div class="upload-card fade-in">
            <div class="card-title">
                <div class="step-number">1</div>
                Upload Resume
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "",
            type="pdf",
            help="📎 Upload your resume in PDF format (Max 10MB)",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            st.success(f"✅ Resume uploaded successfully: **{uploaded_file.name}**")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Step 2: Job Description
        st.markdown("""
        <div class="jd-card fade-in">
            <div class="card-title">
                <div class="step-number">2</div>
                Job Description
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        job_description = st.text_area(
            "",
            height=180,
            placeholder="📋 Paste the complete job description here...\n\nInclude:\n• Required skills and qualifications\n• Job responsibilities\n• Experience requirements\n• Any specific technologies mentioned",
            label_visibility="collapsed"
        )
        
        if job_description.strip():
            word_count = len(job_description.split())
            st.info(f"📊 Job description: **{word_count} words** • {'✅ Good length' if word_count > 50 else '⚠️ Consider adding more details'}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Step 3: Analyze
        st.markdown("""
        <div class="analyze-card fade-in">
            <div class="card-title">
                <div class="step-number">3</div>
                AI Analysis
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Center the analyze button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button(
                "🚀 Analyze with AI",
                type="primary",
                use_container_width=True,
                disabled=not (uploaded_file and job_description.strip())
            )
    
    # Analysis Results
    if analyze_button:
        if not uploaded_file:
            st.error("❌ Please upload a resume PDF file")
            return
        
        if not job_description.strip():
            st.error("❌ Please enter a job description")
            return
        
        # Analysis progress with enhanced UI
        with st.container():
            st.markdown("""
            <div class="results-container fade-in">
                <h3 style="text-align: center; color: #2c3e50; margin-bottom: 2rem;">🤖 AI Analysis in Progress</h3>
            </div>
            """, unsafe_allow_html=True)
            
            progress_container = st.container()
            status_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                
            with status_container:
                status_text = st.empty()
            
            try:
                # Initialize PDF extractor
                pdf_extractor = PDFExtractor()
                
                # Step 1: Extract text
                status_text.markdown("📄 **Extracting text from PDF...**")
                progress_bar.progress(25)
                time.sleep(0.5)
                
                resume_text = pdf_extractor.extract_text(uploaded_file)
                
                # Step 2: Load model
                status_text.markdown("🧠 **Loading AI model...**")
                progress_bar.progress(50)
                time.sleep(0.5)
                
                # Step 3: Analyze
                status_text.markdown("🔍 **Performing semantic analysis...**")
                progress_bar.progress(75)
                time.sleep(0.5)
                
                percentage_score = calculate_semantic_similarity(resume_text, job_description)
                
                # Step 4: Complete
                status_text.markdown("✅ **Analysis complete!**")
                progress_bar.progress(100)
                time.sleep(0.5)
                
                # Clear progress indicators
                progress_container.empty()
                status_container.empty()
                
                # Display Results with enhanced styling
                feedback_message, feedback_type, color, bg_color = get_score_feedback(percentage_score)
                
                st.markdown("""
                <div class="results-container fade-in">
                    <h2 style="text-align: center; color: #2c3e50; margin-bottom: 2rem;">📊 Analysis Results</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Score display with enhanced animation - Fixed f-string issue
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    score_html = f"""
                        <div style="
                            text-align: center;
                            padding: 3rem 2rem;
                            border: 3px solid {color};
                            border-radius: 20px;
                            background: linear-gradient(135deg, {bg_color} 0%, white 100%);
                            margin: 2rem 0;
                            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                            animation: fadeInUp 0.8s ease-out;
                        ">
                            <div class="pulse-animation" style="
                                color: {color};
                                font-size: 5rem;
                                font-weight: 800;
                                margin: 0;
                                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
                            ">{percentage_score}%</div>
                            <h3 style="
                                color: #2c3e50;
                                margin: 1rem 0 0 0;
                                font-weight: 600;
                                font-size: 1.5rem;
                            ">Semantic Match Score</h3>
                            <p style="
                                color: #6c757d;
                                margin: 0.5rem 0 0 0;
                                font-size: 1rem;
                            ">Powered by AI sentence transformers</p>
                        </div>
                    """
                    
                    st.markdown(score_html, unsafe_allow_html=True)
                
                # Feedback message with better styling
                if feedback_type == "success":
                    st.success(feedback_message)
                elif feedback_type == "info":
                    st.info(feedback_message)
                elif feedback_type == "warning":
                    st.warning(feedback_message)
                else:
                    st.error(feedback_message)
                
                # Enhanced analysis details
                with st.expander("📋 **View Detailed Analysis**", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🔧 Technical Details**")
                        st.markdown(f"• **AI Model:** sentence-transformers/all-mpnet-base-v2")
                        st.markdown(f"• **Analysis Method:** Semantic similarity")
                        st.markdown(f"• **Processing Time:** ~2-3 seconds")
                        
                        st.markdown("**📊 Document Stats**")
                        st.markdown(f"• **Resume Words:** {len(resume_text.split())}")
                        st.markdown(f"• **JD Words:** {len(job_description.split())}")
                    
                    with col2:
                        st.markdown("**🎯 Score Breakdown**")
                        if percentage_score >= 80:
                            st.markdown("• **Excellent match** - Apply with confidence")
                        elif percentage_score >= 65:
                            st.markdown("• **Good match** - Highlight key skills")
                        elif percentage_score >= 45:
                            st.markdown("• **Moderate match** - Tailor resume")
                        else:
                            st.markdown("• **Low match** - Consider major changes")
                        
                        # Extract some keywords
                        sections = extract_key_sections(resume_text)
                        if sections['skills']:
                            st.markdown(f"• **Skills Found:** {len(sections['skills'])} sections")
                
            except Exception as e:
                st.error(f"❌ **Analysis Error:** {str(e)}")
                st.info("💡 **Troubleshooting Tips:**\n- Ensure PDF is not password protected\n- Check internet connection for model download\n- Try with a smaller file")
    
    # Enhanced Footer
    st.markdown("""
    <div class="footer">
        <p>🚀 <strong>Powered by</strong> 🤗 HuggingFace • 🧠 Sentence Transformers • ⚡ Streamlit</p>
        <p>Built with ❤️ for better career opportunities</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()