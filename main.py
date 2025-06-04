# main.py
"""Main application entry point for CrewAI Multi-Agent Translation Workflow"""

from workflow import TranslationWorkflow

def print_header():
    """Print application header"""
    print("🤖 === CrewAI Multi-Agent Translation Workflow ===")
    print("🎯 Features: Context Analysis | Translation Refinement | Quality Assessment")
    print("👥 Agents: Professional Translator | Cultural Context Analyst | Quality Assessor")
    print("\nCommands: 'single', 'batch', 'quit'\n")

def print_translation_result(result):
    """Print formatted translation result"""
    print("\n" + "="*60)
    print("🎯 TRANSLATION RESULT")
    print("="*60)
    print(f"📝 Original: {result['original_text']}")
    print(f"🔄 Initial: {result['initial_translation']}")
    print(f"✨ Refined: {result['refined_translation']}")
    print(f"📊 Scores: {result['accuracy_score']}")
    print(f"📋 Assessment: {result['quality_assessment']}")
    print(f"🎯 Final Recommendation: {result['final_recommendation']}")
    print(f"🧠 Context Analysis: {result['context_analysis']}")
    print("="*60)

def print_batch_results(results):
    """Print formatted batch translation results"""
    print("\n" + "="*60)
    print("📊 BATCH TRANSLATION RESULTS")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        print(f"\n🔍 --- TRANSLATION {i} ---")
        print(f"📝 Original: {result['original_text']}")
        print(f"🔄 Initial: {result['initial_translation']}")
        print(f"✨ Refined: {result['refined_translation']}")
        print(f"📊 Scores: {result['accuracy_score']}")
        print(f"🎯 Context: {result['context_analysis'][:150]}...")

def get_user_input():
    """Get and validate user input"""
    text = input("📝 Text to translate: ").strip()
    if not text:
        return None, None, None
    
    source = input("🔤 From language [en]: ").strip() or "en"
    target = input("🎯 To language [fr]: ").strip() or "fr"
    
    return text, source, target

def get_batch_input():
    """Get batch translation input"""
    texts = []
    print("Enter texts (empty line to finish):")
    
    while True:
        text = input(f"Text {len(texts)+1}: ").strip()
        if not text:
            break
        texts.append(text)
    
    if not texts:
        return None, None, None
    
    source = input("From language [en]: ").strip() or "en"
    target = input("To language [fr]: ").strip() or "fr"
    
    return texts, source, target

def run_single_mode(workflow):
    """Run single translation mode"""
    text, source, target = get_user_input()
    if not text:
        return
    
    result = workflow.translate(text, source, target)
    print_translation_result(result)

def run_batch_mode(workflow):
    """Run batch translation mode"""
    print("\n📦 Batch Translation Mode")
    texts, source, target = get_batch_input()
    
    if not texts:
        print("No texts provided.")
        return
    
    results = workflow.batch_translate(texts, source, target)
    print_batch_results(results)

def main():
    """Main application function"""
    print_header()
    
    # Initialize workflow
    try:
        workflow = TranslationWorkflow()
    except Exception as e:
        print(f"❌ Failed to initialize workflow: {e}")
        print("💡 Make sure you have:")
        print("   - Valid Gemini API key")
        print("   - Internet connection for model downloads")
        print("   - Required packages: pip install crewai transformers torch google-generativeai")
        return
    
    # Main interaction loop
    while True:
        try:
            mode = input("🎮 Mode [single/batch/quit]: ").strip().lower()
            
            if mode == 'quit':
                print("👋 Goodbye!")
                break
            elif mode == 'batch':
                run_batch_mode(workflow)
            elif mode == 'single' or mode == '':
                run_single_mode(workflow)
            else:
                print("❓ Please enter 'single', 'batch', or 'quit'")
                
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Try again or type 'quit' to exit")

if __name__ == "__main__":
    main()