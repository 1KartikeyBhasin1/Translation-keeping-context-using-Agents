# workflow.py
"""Core translation workflow with M2M100 and multi-agent processing"""

from typing import Dict, TypedDict, List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from crewai import Task, Crew, Process
from agents import TranslationAgents

# Configuration
TRANSLATION_MODEL = "facebook/m2m100_418M"

class TranslationResult(TypedDict):
    """Structure for translation results"""
    original_text: str
    source_lang: str
    target_lang: str
    initial_translation: str
    context_analysis: str
    refined_translation: str
    accuracy_score: str
    quality_assessment: str
    final_recommendation: str

class TranslationWorkflow:
    """Main workflow class that orchestrates translation process"""
    
    def __init__(self):
        # Initialize translation model
        print("🔄 Loading translation model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL)
            print("✅ Translation model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load translation model: {e}")
            raise
        
        # Initialize agents
        try:
            self.agent_factory = TranslationAgents()
            self.translator_agent = self.agent_factory.create_translator_agent()
            self.context_agent = self.agent_factory.create_context_agent()
            self.quality_agent = self.agent_factory.create_quality_agent()
            print("✅ All agents initialized successfully!")
        except Exception as e:
            print(f"❌ Failed to initialize agents: {e}")
            raise
    
    def _get_initial_translation(self, text: str, source_lang: str, target_lang: str) -> str:
        """Get initial translation using M2M100 model"""
        try:
            self.tokenizer.src_lang = source_lang
            inputs = self.tokenizer(text, return_tensors="pt")
            
            # Handle language code mapping
            lang_mapping = {
                'en': 'en', 'fr': 'fr', 'es': 'es', 'de': 'de', 'it': 'it',
                'pt': 'pt', 'ru': 'ru', 'zh': 'zh', 'ja': 'ja', 'ko': 'ko',
                'ar': 'ar', 'hi': 'hi'
            }
            
            mapped_target = lang_mapping.get(target_lang, 'fr')  # Default to French
            
            try:
                inputs["forced_bos_token_id"] = self.tokenizer.get_lang_id(mapped_target)
            except:
                inputs["forced_bos_token_id"] = self.tokenizer.get_lang_id("fr")
                print(f"⚠️ Language '{target_lang}' not supported, using 'fr' as fallback")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_length=200, 
                    num_beams=4, 
                    early_stopping=True,
                    do_sample=False
                )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        except Exception as e:
            print(f"❌ Translation error: {e}")
            return f"Translation failed: {str(e)}"
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        """Execute the complete multi-agent translation workflow"""
        print(f"\n🚀 Starting multi-agent translation: {source_lang} → {target_lang}")
        print(f"📝 Text: '{text}'")
        
        # Step 1: Get initial translation
        print("\n🔄 Getting initial translation...")
        initial_translation = self._get_initial_translation(text, source_lang, target_lang)
        print(f"Initial: {initial_translation}")
        
        # Step 2: Context Analysis Task
        context_task = Task(
            description=f"""
            Analyze the following text for translation context:
            
            Original Text: "{text}"
            Source Language: {source_lang}
            Target Language: {target_lang}
            Initial Translation: "{initial_translation}"
            
            Please provide:
            1. Text type and domain (formal/casual, business/personal, etc.)
            2. Cultural elements that need special attention
            3. Tone and register analysis
            4. Specific challenges for this language pair
            5. Recommendations for cultural adaptation
            
            Format your response as a detailed analysis in 3-4 sentences.
            """,
            agent=self.context_agent,
            expected_output="Detailed context analysis with cultural and linguistic insights"
        )
        
        # Step 3: Translation Refinement Task
        refinement_task = Task(
            description=f"""
            Based on the context analysis, refine this translation:
            
            Original Text: "{text}"
            Initial Translation: "{initial_translation}"
            Target Language: {target_lang}
            
            Create an improved translation that:
            1. Sounds more natural and fluent
            2. Maintains the original meaning and tone
            3. Incorporates cultural appropriateness
            4. Fixes any grammatical or stylistic issues
            5. Matches the appropriate register (formal/informal)
            
            Provide ONLY the refined translation, nothing else.
            """,
            agent=self.translator_agent,
            expected_output="Refined, natural translation that incorporates context insights",
            context=[context_task]
        )
        
        # Step 4: Quality Assessment Task
        quality_task = Task(
            description=f"""
            Evaluate the translation quality and provide comprehensive assessment:
            
            Original ({source_lang}): "{text}"
            Initial Translation: "{initial_translation}"
            Target Language: {target_lang}
            
            Assess and score (1-10):
            1. Accuracy (meaning preservation)
            2. Fluency (natural sound)
            3. Cultural appropriateness
            4. Overall quality
            
            Also provide:
            - Specific strengths and weaknesses
            - Final recommendation (use refined or suggest further improvements)
            - Brief justification for scores
            
            Format as:
            SCORES: Accuracy: X/10, Fluency: X/10, Cultural: X/10, Overall: X/10
            ASSESSMENT: [detailed analysis]
            RECOMMENDATION: [final recommendation]
            """,
            agent=self.quality_agent,
            expected_output="Comprehensive quality assessment with scores and recommendations",
            context=[context_task, refinement_task]
        )
        
        # Create and execute the crew
        try:
            crew = Crew(
                agents=[self.context_agent, self.translator_agent, self.quality_agent],
                tasks=[context_task, refinement_task, quality_task],
                process=Process.sequential,
                verbose=False
            )
            
            print("\n🤖 Executing multi-agent crew...")
            result = crew.kickoff()
            
            # Parse results from task outputs
            context_analysis = str(context_task.output) if hasattr(context_task, 'output') and context_task.output else "Context analysis not available"
            refined_translation = str(refinement_task.output) if hasattr(refinement_task, 'output') and refinement_task.output else initial_translation
            quality_assessment = str(quality_task.output) if hasattr(quality_task, 'output') and quality_task.output else "Quality assessment not available"
            
        except Exception as e:
            print(f"❌ Crew execution failed: {e}")
            # Fallback to initial translation
            context_analysis = "Multi-agent analysis failed"
            refined_translation = initial_translation
            quality_assessment = f"Quality assessment failed: {str(e)}"
        
        # Extract scores and recommendation from quality assessment
        accuracy_score = "N/A"
        final_recommendation = refined_translation
        
        if "SCORES:" in quality_assessment:
            try:
                scores_section = quality_assessment.split("SCORES:")[1].split("ASSESSMENT:")[0]
                accuracy_score = scores_section.strip()
            except:
                pass
        
        if "RECOMMENDATION:" in quality_assessment:
            try:
                final_recommendation = quality_assessment.split("RECOMMENDATION:")[1].strip()
            except:
                pass
        
        return TranslationResult(
            original_text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            initial_translation=initial_translation,
            context_analysis=context_analysis,
            refined_translation=refined_translation,
            accuracy_score=accuracy_score,
            quality_assessment=quality_assessment,
            final_recommendation=final_recommendation
        )
    
    def batch_translate(self, texts: List[str], source_lang: str, target_lang: str) -> List[TranslationResult]:
        """Process multiple texts with the multi-agent workflow"""
        results = []
        for i, text in enumerate(texts, 1):
            print(f"\n📋 Processing batch item {i}/{len(texts)}")
            result = self.translate(text, source_lang, target_lang)
            results.append(result)
        return results