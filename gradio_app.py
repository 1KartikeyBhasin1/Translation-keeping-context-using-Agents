# gradio_app.py
"""Gradio frontend for CrewAI Multi-Agent Translation Workflow"""

import gradio as gr
import pandas as pd
from typing import List, Tuple
from workflow import TranslationWorkflow

# Language mappings with display names and codes
LANGUAGES = {
    "English": "en",
    "French": "fr", 
    "Spanish": "es",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Hindi": "hi",
    "Dutch": "nl",
    "Turkish": "tr",
    "Polish": "pl"
}

class GradioTranslationApp:
    def __init__(self):
        """Initialize the Gradio app with workflow"""
        self.workflow = None
        self.initialize_workflow()
    
    def initialize_workflow(self):
        """Initialize the translation workflow"""
        try:
            self.workflow = TranslationWorkflow()
            return "✅ Translation workflow initialized successfully!"
        except Exception as e:
            return f"❌ Failed to initialize workflow: {str(e)}"
    
    def translate_single(self, text: str, source_lang: str, target_lang: str) -> Tuple[str, str, str, str, str, str]:
        """Handle single text translation"""
        if not self.workflow:
            error_msg = "❌ Workflow not initialized. Please check your setup."
            return error_msg, "", "", "", "", ""
        
        if not text.strip():
            return "⚠️ Please enter text to translate", "", "", "", "", ""
        
        if source_lang == target_lang:
            return "⚠️ Source and target languages cannot be the same", "", "", "", "", ""
        
        try:
            # Get language codes
            source_code = LANGUAGES[source_lang]
            target_code = LANGUAGES[target_lang]
            
            # Perform translation
            result = self.workflow.translate(text, source_code, target_code)
            
            # Format results
            status = f"✅ Translation completed: {source_lang} → {target_lang}"
            initial = result['initial_translation']
            refined = result['refined_translation']
            context = result['context_analysis']
            scores = result['accuracy_score']
            assessment = result['quality_assessment']
            
            return status, initial, refined, context, scores, assessment
            
        except Exception as e:
            error_msg = f"❌ Translation failed: {str(e)}"
            return error_msg, "", "", "", "", ""
    
    def translate_batch(self, batch_text: str, source_lang: str, target_lang: str) -> Tuple[str, pd.DataFrame]:
        """Handle batch text translation"""
        if not self.workflow:
            return "❌ Workflow not initialized. Please check your setup.", pd.DataFrame()
        
        if not batch_text.strip():
            return "⚠️ Please enter texts to translate (one per line)", pd.DataFrame()
        
        if source_lang == target_lang:
            return "⚠️ Source and target languages cannot be the same", pd.DataFrame()
        
        try:
            # Split texts by lines
            texts = [line.strip() for line in batch_text.split('\n') if line.strip()]
            
            if not texts:
                return "⚠️ No valid texts found", pd.DataFrame()
            
            # Get language codes
            source_code = LANGUAGES[source_lang]
            target_code = LANGUAGES[target_lang]
            
            # Perform batch translation
            results = self.workflow.batch_translate(texts, source_code, target_code)
            
            # Create DataFrame for results
            df_data = []
            for i, result in enumerate(results, 1):
                df_data.append({
                    "Item": i,
                    "Original Text": result['original_text'],
                    "Initial Translation": result['initial_translation'],
                    "Refined Translation": result['refined_translation'],
                    "Context Analysis": result['context_analysis'][:100] + "..." if len(result['context_analysis']) > 100 else result['context_analysis'],
                    "Quality Scores": result['accuracy_score']
                })
            
            df = pd.DataFrame(df_data)
            status = f"✅ Batch translation completed: {len(texts)} texts processed ({source_lang} → {target_lang})"
            
            return status, df
            
        except Exception as e:
            error_msg = f"❌ Batch translation failed: {str(e)}"
            return error_msg, pd.DataFrame()
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        # Custom CSS for better styling
        css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .header {
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .status-box {
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        """
        
        with gr.Blocks(css=css, title="🤖 Multi-Agent Translation Workflow") as app:
            
            # Header
            gr.HTML("""
            <div class="header">
                <h1>🤖 CrewAI Multi-Agent Translation Workflow</h1>
                <p>🎯 Features: Context Analysis | Translation Refinement | Quality Assessment</p>
                <p>👥 Agents: Professional Translator | Cultural Context Analyst | Quality Assessor</p>
            </div>
            """)
            
            # Initialization status
            with gr.Row():
                init_status = gr.Textbox(
                    value=self.initialize_workflow(),
                    label="🔧 System Status",
                    interactive=False,
                    show_copy_button=True
                )
            
            # Main interface with tabs
            with gr.Tabs():
                
                # Single Translation Tab
                with gr.TabItem("🔤 Single Translation"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            single_text = gr.Textbox(
                                label="📝 Text to Translate",
                                placeholder="Enter the text you want to translate...",
                                lines=4,
                                max_lines=10
                            )
                        
                        with gr.Column(scale=1):
                            single_source = gr.Dropdown(
                                choices=list(LANGUAGES.keys()),
                                value="English",
                                label="🔤 From Language"
                            )
                            single_target = gr.Dropdown(
                                choices=list(LANGUAGES.keys()),
                                value="French",
                                label="🎯 To Language"
                            )
                            translate_btn = gr.Button("🚀 Translate", variant="primary", size="lg")
                    
                    # Results section
                    single_status = gr.Textbox(label="📊 Status", interactive=False)
                    
                    with gr.Row():
                        with gr.Column():
                            initial_translation = gr.Textbox(
                                label="🔄 Initial Translation",
                                interactive=False,
                                lines=3
                            )
                            refined_translation = gr.Textbox(
                                label="✨ Refined Translation",
                                interactive=False,
                                lines=3
                            )
                        
                        with gr.Column():
                            context_analysis = gr.Textbox(
                                label="🧠 Context Analysis",
                                interactive=False,
                                lines=4
                            )
                            quality_scores = gr.Textbox(
                                label="📊 Quality Scores",
                                interactive=False,
                                lines=2
                            )
                    
                    quality_assessment = gr.Textbox(
                        label="📋 Quality Assessment",
                        interactive=False,
                        lines=4
                    )
                
                # Batch Translation Tab
                with gr.TabItem("📦 Batch Translation"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            batch_text = gr.Textbox(
                                label="📝 Texts to Translate (one per line)",
                                placeholder="Enter multiple texts, one per line:\nHello world\nHow are you?\nGoodbye",
                                lines=8,
                                max_lines=15
                            )
                        
                        with gr.Column(scale=1):
                            batch_source = gr.Dropdown(
                                choices=list(LANGUAGES.keys()),
                                value="English",
                                label="🔤 From Language"
                            )
                            batch_target = gr.Dropdown(
                                choices=list(LANGUAGES.keys()),
                                value="Spanish",
                                label="🎯 To Language"
                            )
                            batch_translate_btn = gr.Button("📦 Batch Translate", variant="primary", size="lg")
                    
                    # Batch results
                    batch_status = gr.Textbox(label="📊 Batch Status", interactive=False)
                    batch_results = gr.Dataframe(
                        label="📋 Translation Results",
                        interactive=False,
                        wrap=True
                    )
                
                # About Tab
                with gr.TabItem("ℹ️ About"):
                    gr.Markdown("""
                    ## 🤖 Multi-Agent Translation Workflow
                    
                    This application uses a sophisticated multi-agent system powered by CrewAI to provide high-quality translations:
                    
                    ### 👥 Meet the Agents:
                    - **🔤 Professional Translator**: Provides initial translations and refinements
                    - **🧠 Cultural Context Analyst**: Analyzes cultural nuances and context
                    - **📊 Quality Assessor**: Evaluates translation quality and provides recommendations
                    
                    ### 🎯 Features:
                    - **Context-Aware Translation**: Considers cultural and linguistic context
                    - **Multi-Stage Process**: Initial translation → Context analysis → Refinement → Quality assessment
                    - **Batch Processing**: Translate multiple texts at once
                    - **Quality Scoring**: Detailed assessment with accuracy and fluency scores
                    
                    ### 🔧 Technology Stack:
                    - **Translation Engine**: Facebook M2M100 (multilingual machine translation)
                    - **AI Agents**: CrewAI with Gemini 2.0 Flash
                    - **Frontend**: Gradio
                    - **Languages Supported**: 15+ languages including major European, Asian, and Middle Eastern languages
                    
                    ### 💡 Tips for Best Results:
                    1. Use clear, well-structured sentences
                    2. Provide context when translating ambiguous terms
                    3. Review the context analysis for cultural insights
                    4. Consider the refined translation for final use
                    """)
            
            # Event handlers
            translate_btn.click(
                fn=self.translate_single,
                inputs=[single_text, single_source, single_target],
                outputs=[single_status, initial_translation, refined_translation, 
                        context_analysis, quality_scores, quality_assessment]
            )
            
            batch_translate_btn.click(
                fn=self.translate_batch,
                inputs=[batch_text, batch_source, batch_target],
                outputs=[batch_status, batch_results]
            )
            
            # Examples
            gr.Examples(
                examples=[
                    ["Hello, how are you today?", "English", "French"],
                    ["I love programming and artificial intelligence.", "English", "Spanish"],
                    ["The weather is beautiful today.", "English", "German"],
                    ["Thank you for your help!", "English", "Japanese"],
                ],
                inputs=[single_text, single_source, single_target],
                label="📝 Try these examples"
            )
        
        return app

def main():
    """Main function to run the Gradio app"""
    app_instance = GradioTranslationApp()
    app = app_instance.create_interface()
    
    # Launch the app
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create a public link
        show_error=True         # Show errors in the interface
    )

if __name__ == "__main__":
    main()