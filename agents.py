# agents.py
"""Agent classes and LLM wrapper for the translation workflow"""

from typing import Optional, List
import google.generativeai as genai
from crewai import Agent
from crewai.llm import LLM

# Configuration
GEMINI_API_KEY = "AIzaSyALkgl1UdmkvcSVXK3j9CYrKp6Um_Fxc1A"  # Replace with your API key
GEMINI_MODEL_NAME = "gemini-1.5-flash-002"

class GeminiLLM(LLM):
    """Proper CrewAI LLM wrapper for Gemini"""
    
    def __init__(self, api_key: str, model_name: str = GEMINI_MODEL_NAME, **kwargs):
        # Pass the model name to the parent LLM class
        super().__init__(model=model_name, **kwargs)
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        print(f"✅ Initialized Gemini LLM: {model_name}")
    
    def supports_stop_words(self) -> bool:
        return False
    
    def call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Main method called by CrewAI agents"""
        try:
            # Handle different input formats that CrewAI might send
            if isinstance(prompt, str):
                # Simple string prompt
                content = prompt
            elif isinstance(prompt, list):
                # List of messages - extract content
                content = ""
                for msg in prompt:
                    if isinstance(msg, dict):
                        if 'content' in msg:
                            content += msg['content'] + "\n"
                        elif 'text' in msg:
                            content += msg['text'] + "\n"
                    elif isinstance(msg, str):
                        content += msg + "\n"
                content = content.strip()
            elif isinstance(prompt, dict):
                # Single message dict
                if 'content' in prompt:
                    content = prompt['content']
                elif 'text' in prompt:
                    content = prompt['text']
                else:
                    content = str(prompt)
            else:
                # Fallback to string conversion
                content = str(prompt)
            
            # Generate response using Gemini
            response = self.gemini_model.generate_content(content)
            return response.text if response.text else "No response generated"
            
        except Exception as e:
            print(f"❌ Gemini API Error: {str(e)}")
            # Return a more helpful error message
            return f"I apologize, but I encountered an error while processing your request. Please try again with a simpler prompt."
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Fallback call method"""
        return self.call(prompt, **kwargs)


class TranslationAgents:
    """Factory class for creating translation agents"""
    
    def __init__(self, api_key: str = GEMINI_API_KEY):
        """Initialize with Gemini LLM"""
        self.llm = GeminiLLM(api_key=api_key)
    
    def create_translator_agent(self) -> Agent:
        """Agent responsible for initial translation and refinement"""
        return Agent(
            role="Professional Translator",
            goal="Provide accurate and natural translations while maintaining the original meaning and tone",
            backstory="""You are an expert linguist with deep knowledge of multiple languages. 
            You specialize in creating natural, culturally appropriate translations that preserve 
            the original intent while being fluent in the target language. You have experience 
            with various text types from casual conversation to technical documents.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm
        )
    
    def create_context_agent(self) -> Agent:
        """Agent responsible for context analysis and cultural adaptation"""
        return Agent(
            role="Cultural Context Analyst",
            goal="Analyze text context, cultural nuances, and provide recommendations for optimal translation",
            backstory="""You are a cultural linguistics expert who specializes in understanding 
            the subtle cultural, social, and contextual elements in text. You can identify 
            formal vs informal registers, cultural references, idioms, and suggest how to 
            best adapt them for different target cultures while maintaining authenticity.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm
        )
    
    def create_quality_agent(self) -> Agent:
        """Agent responsible for quality assessment and final recommendations"""
        return Agent(
            role="Translation Quality Assessor",
            goal="Evaluate translation accuracy, fluency, and cultural appropriateness to ensure highest quality output",
            backstory="""You are a meticulous quality control expert in translation services. 
            You have extensive experience in evaluating translations across multiple criteria 
            including accuracy, fluency, cultural appropriateness, and naturalness. You provide 
            detailed feedback and actionable recommendations for improvement.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm
        )