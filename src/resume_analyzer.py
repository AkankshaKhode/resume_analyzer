# from langchain.llms import OpenAI                    # Commented out - OpenAI
# from langchain.chat_models import ChatOpenAI         # Commented out - OpenAI
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any
import json
import re
from .config import Config

class ResumeAnalyzer:
    """Main resume analysis class using LangChain and HuggingFace models"""
    
    def __init__(self, config: Config):
        self.config = config
        
        if config.provider == 'huggingface':
            self.llm = self._setup_huggingface_llm()
        # elif config.provider == 'openai':              # Commented out - OpenAI
        #     self.llm = self._setup_openai_llm()         # Commented out - OpenAI
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
        
        # Define the analysis prompt (optimized for HuggingFace models)
        self.analysis_prompt = PromptTemplate(
            input_variables=["resume_text", "job_description"],
            template="""
            Analyze this resume against the job description. Provide specific feedback.
            
            Resume: {resume_text}
            
            Job Description: {job_description}
            
            Analysis:
            Skills Match: [List matching skills]
            Missing Skills: [List missing important skills]
            Improvement Tips: [3-5 specific actionable tips]
            Overall Score: [Score 1-10]
            Strengths: [Key candidate strengths]
            Weaknesses: [Areas for improvement]
            """
        )
        
        # Create the analysis chain
        self.analysis_chain = LLMChain(
            llm=self.llm,
            prompt=self.analysis_prompt
        )
    
    def _setup_huggingface_llm(self) -> HuggingFacePipeline:
        """Setup HuggingFace LLM pipeline"""
        try:
            # Create text generation pipeline
            hf_pipeline = pipeline(
                "text-generation",
                model=self.config.hf_model_name,
                tokenizer=self.config.hf_model_name,
                max_length=self.config.hf_max_length,
                temperature=self.config.hf_temperature,
                device_map=self.config.hf_device,
                return_full_text=False,
                do_sample=True,
                pad_token_id=50256  # Common pad token for GPT-style models
            )
            
            # Wrap in LangChain
            llm = HuggingFacePipeline(pipeline=hf_pipeline)
            return llm
            
        except Exception as e:
            print(f"Error setting up HuggingFace model: {e}")
            # Fallback to a smaller, more reliable model
            print("Falling back to distilgpt2...")
            hf_pipeline = pipeline(
                "text-generation",
                model="distilgpt2",
                max_length=512,
                temperature=0.7,
                return_full_text=False,
                do_sample=True
            )
            return HuggingFacePipeline(pipeline=hf_pipeline)
    
    # Commented out - OpenAI setup
    # def _setup_openai_llm(self) -> ChatOpenAI:
    #     """Setup OpenAI LLM"""
    #     return ChatOpenAI(
    #         openai_api_key=self.config.api_key,
    #         model_name=self.config.openai_model,
    #         temperature=self.config.openai_temperature,
    #         max_tokens=self.config.openai_max_tokens
    #     )
    
    def analyze_resume(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """
        Analyze resume against job description
        
        Args:
            resume_text: Extracted text from resume PDF
            job_description: Job description text
            
        Returns:
            Dict containing analysis results
        """
        try:
            # Truncate inputs if too long (HuggingFace models have token limits)
            resume_text = resume_text[:2000] if len(resume_text) > 2000 else resume_text
            job_description = job_description[:1000] if len(job_description) > 1000 else job_description
            
            # Run the analysis
            result = self.analysis_chain.run(
                resume_text=resume_text,
                job_description=job_description
            )
            
            # Parse the structured response
            analysis = self._parse_analysis_result(result)
            return analysis
                
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return {
                "skills_match": "Analysis encountered an error. Please try again.",
                "missing_skills": "Unable to determine missing skills.", 
                "improvement_tips": f"Error during analysis: {str(e)}",
                "overall_score": "N/A",
                "strengths": "Analysis incomplete",
                "weaknesses": "Analysis incomplete"
            }
    
    def _parse_analysis_result(self, result: str) -> Dict[str, Any]:
        """Parse the LLM result into structured format"""
        try:
            # Try to extract structured information using regex
            analysis = {}
            
            # Extract each section
            skills_match = re.search(r'Skills Match:(.+?)(?=Missing Skills:|$)', result, re.DOTALL)
            missing_skills = re.search(r'Missing Skills:(.+?)(?=Improvement Tips:|$)', result, re.DOTALL)
            improvement_tips = re.search(r'Improvement Tips:(.+?)(?=Overall Score:|$)', result, re.DOTALL)
            overall_score = re.search(r'Overall Score:(.+?)(?=Strengths:|$)', result, re.DOTALL)
            strengths = re.search(r'Strengths:(.+?)(?=Weaknesses:|$)', result, re.DOTALL)
            weaknesses = re.search(r'Weaknesses:(.+?)$', result, re.DOTALL)
            
            analysis['skills_match'] = skills_match.group(1).strip() if skills_match else "Analysis completed"
            analysis['missing_skills'] = missing_skills.group(1).strip() if missing_skills else "See full analysis"
            analysis['improvement_tips'] = improvement_tips.group(1).strip() if improvement_tips else result[:500]
            analysis['overall_score'] = overall_score.group(1).strip() if overall_score else "N/A"
            analysis['strengths'] = strengths.group(1).strip() if strengths else "Analysis completed"
            analysis['weaknesses'] = weaknesses.group(1).strip() if weaknesses else "See improvement tips"
            
            return analysis
            
        except Exception:
            # Fallback: return the raw result in improvement_tips
            return {
                "skills_match": "Analysis completed - see detailed feedback below",
                "missing_skills": "Analysis completed - see detailed feedback below", 
                "improvement_tips": result,
                "overall_score": "N/A",
                "strengths": "Analysis completed",
                "weaknesses": "See detailed feedback"
            }
    
    def extract_skills(self, text: str) -> list:
        """Extract skills from text using LLM"""
        skills_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Extract technical skills, soft skills, and qualifications from this text.
            List them separated by commas.
            
            Text: {text}
            
            Skills:
            """
        )
        
        skills_chain = LLMChain(llm=self.llm, prompt=skills_prompt)
        result = skills_chain.run(text=text[:1000])  # Limit input length
        
        # Parse skills from result
        skills = [skill.strip() for skill in result.replace('\n', ',').split(',') if skill.strip()]
        return skills[:20]  # Limit to 20 skills